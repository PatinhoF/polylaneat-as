"""
COCOLaneDataset
================

Native loader for lane-marking datasets annotated in COCO segmentation
format (e.g. exported from Roboflow/CVAT), WITHOUT converting them to
the TuSimple format first.

Design note
-----------
This class follows the exact same low-level contract used by every other
raw dataset loader in this package (see lib/datasets/tusimple.py,
lib/datasets/elas.py, lib/datasets/culane.py, lib/datasets/llamas.py):

    * self.annotations  -> list[dict] with keys 'path' and 'lanes', where
                            'lanes' is a list of lane point lists
                            [(x, y), (x, y), ...] in pixel coordinates of
                            the *original* image.
    * self.max_lanes     -> largest number of lanes found in one image,
                             across the whole (possibly merged) dataset.
    * self.max_points    -> largest number of points found in one lane.
    * get_img_heigth(path) / get_img_width(path)
                          -> original per-image dimensions. Called by
                             PointsDataset (lib/datasets/lane_dataset.py)
                             to rescale points to the network's input
                             resolution BEFORE it fits the polynomial.
    * get_metrics(...) / eval(...)
                          -> placeholders, following the same convention
                             already used by lib/datasets/elas.py, since
                             there is no official benchmark metric defined
                             for a private/custom dataset.
    * __getitem__ / __len__

Because PointsDataset is completely agnostic to *how* the raw lane points
were produced, none of the following is reimplemented here -- it is fully
inherited, UNMODIFIED, from PointsDataset.transform_annotation():
    - the degree-3 polynomial fit (np.polyfit)
    - the upper/lower y-limit computation
    - the imgaug augmentation pipeline
    - the final fixed-size target tensor expected by the model/loss

This file's only job is: COCO annotations in -> ordered (x, y) lane point
lists out.
"""
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


class COCOLaneDataset(object):
    """Loads one or several COCO-segmentation lane datasets into a single
    flat list of annotations, compatible with PolyLaneNet's PointsDataset.

    Supports two usage modes:

    1) Multiple datasets, concatenated into one (used for the training /
       validation splits described in the task, e.g. apagadas1 + dia1 +
       noite1 + nevoeiro1):

           COCOLaneDataset(datasets=[
               {'root': 'datasets/apagadas1', 'annotation': 'annotations/train.json'},
               {'root': 'datasets/dia1',      'annotation': 'annotations/train.json'},
           ])

    2) A single dataset, for convenience/backwards-compatibility with the
       root/annotation style used by the other loaders:

           COCOLaneDataset(root='datasets/apagadas1', annotation='annotations/train.json')
    """

    def __init__(self,
                 datasets=None,
                 root=None,
                 annotation=None,
                 category_id=1,
                 min_points=3,
                 max_lanes=None,
                 images_dir=None):
        """
        Args:
            datasets: list[dict] each with keys 'root' and 'annotation'
                (and, optionally, 'images_dir'). When given, all listed
                COCO datasets are loaded and concatenated into a single
                flat `self.annotations` list -- this is what makes
                "multiple datasets" behave as one PyTorch Dataset.
            root / annotation: convenience params to load a *single* COCO
                dataset without wrapping it inside `datasets`. Ignored if
                `datasets` is provided.
            category_id: COCO category_id that corresponds to a lane
                marking (the task specifies category_id = 1). Annotations
                with any other category_id are ignored.
            min_points: minimum number of centerline points a decoded
                lane mask must yield to be kept. Lanes with fewer points
                (degenerate/too-thin masks, decode failures, etc.) are
                silently discarded -- this is the "ignore invalid lanes
                automatically" requirement.
            max_lanes: optional override, exactly like in tusimple.py /
                elas.py / llamas.py. Useful to force compatibility with a
                model whose `num_outputs` was sized for a fixed number of
                lane "slots".
            images_dir: optional override for locating image files -- see
                `_resolve_image_path` below for the default search order.
        """
        if datasets is None:
            if root is None or annotation is None:
                raise Exception('Either `datasets` or both `root` and `annotation` must be specified')
            datasets = [{'root': root, 'annotation': annotation, 'images_dir': images_dir}]

        self.datasets = datasets
        self.category_id = category_id
        self.min_points = min_points

        self.load_annotations()

        # Force max_lanes, used when the model/config expects a fixed
        # number of lane slots (same convention as tusimple.py/elas.py).
        if max_lanes is not None:
            self.max_lanes = max_lanes

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------
    def load_annotations(self):
        self.annotations = []
        self.max_lanes = 0
        self.max_points = 0
        self._img_sizes = {}  # path -> (width, height), used by get_img_width/heigth

        for dataset_cfg in self.datasets:
            self._load_single_coco_dataset(dataset_cfg)

        print('{} annotations found. max_points: {} | max_lanes: {}'.format(
            len(self.annotations), self.max_points, self.max_lanes))

    def _load_single_coco_dataset(self, dataset_cfg):
        root = dataset_cfg['root']
        annotation_path = dataset_cfg['annotation']
        images_dir = dataset_cfg.get('images_dir')

        if not os.path.isabs(annotation_path):
            annotation_path = os.path.join(root, annotation_path)
        if not os.path.isfile(annotation_path):
            raise Exception('COCO annotation file not found: {}'.format(annotation_path))

        coco = COCO(annotation_path)

        for img_id, img_info in coco.imgs.items():
            img_path = self._resolve_image_path(root, annotation_path, images_dir, img_info['file_name'])
            img_w = img_info.get('width')
            img_h = img_info.get('height')

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[self.category_id])
            anns = coco.loadAnns(ann_ids)

            lanes = []
            for ann in anns:
                points = self._annotation_to_points(ann)
                if points is None:
                    continue  # invalid lane: ignored automatically
                lanes.append(points)
                self.max_points = max(self.max_points, len(points))

            if len(lanes) == 0:
                # No valid lane in this image: skip it. (Change this to
                # `lanes = []` + keep appending if you want to train on
                # lane-free negative images as well.)
                continue

            self.max_lanes = max(self.max_lanes, len(lanes))
            self._img_sizes[img_path] = (img_w, img_h)
            self.annotations.append({
                'lanes': lanes,
                'path': img_path,
                'categories': [1] * len(lanes),
            })

    def _resolve_image_path(self, root, annotation_path, images_dir, file_name):
        """Resolves the on-disk path of an image referenced in a COCO json.

        Tries, in order:
            1. `images_dir` explicitly given for this dataset entry.
            2. `root/file_name` (file_name already relative to the dataset
               root -- the layout used by most COCO/Roboflow exports).
            3. The directory containing the annotation file itself (common
               when images sit next to `_annotations.coco.json`).

        NOTE: if your folder layout doesn't match any of the above,
        adjust this single method -- nothing else needs to change.
        """
        candidates = []
        if images_dir is not None:
            candidates.append(os.path.join(images_dir, file_name))
        candidates.append(os.path.join(root, file_name))
        candidates.append(os.path.join(os.path.dirname(annotation_path), file_name))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        # Fall back to the most standard candidate even if not found yet;
        # a clear FileNotFoundError will surface later when the image is
        # actually read, which is easier to debug than a silent wrong path.
        return candidates[1] if images_dir is None else candidates[0]

    # ------------------------------------------------------------------
    # RLE mask -> ordered centerline points
    # ------------------------------------------------------------------
    def _annotation_to_points(self, ann):
        """Decodes a compressed COCO RLE mask and extracts an ordered
        (x, y) centerline for the lane, cropped to the annotation's
        bounding box for both correctness and speed.
        """
        rle = ann.get('segmentation')
        if not rle or 'counts' not in rle or 'size' not in rle:
            return None

        rle = dict(rle)  # don't mutate the original annotation dict
        if isinstance(rle['counts'], str):
            # pycocotools requires bytes for compressed ("compressed RLE") counts;
            # json deserializes them as str, so re-encode before decoding.
            rle['counts'] = rle['counts'].encode('utf-8')

        try:
            mask = mask_utils.decode(rle)
        except Exception:
            return None  # corrupted/invalid RLE: ignore this lane

        if mask is None or mask.sum() == 0:
            return None

        bbox = ann.get('bbox', [0, 0, mask.shape[1], mask.shape[0]])
        x, y, w, h = bbox
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + w)), mask.shape[1]), min(int(round(y + h)), mask.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None  # degenerate bounding box: ignore this lane

        cropped_mask = mask[y0:y1, x0:x1]
        points = self._centerline_from_mask(cropped_mask, x_offset=x0, y_offset=y0)
        if points is None or len(points) < self.min_points:
            return None
        return points

    @staticmethod
    def _centerline_from_mask(mask, x_offset, y_offset):
        """Extracts an ordered (x, y) centerline by averaging, for every
        row of the mask, the x coordinates of its foreground pixels.

        This mirrors the "one x sample per y row" representation used by
        TuSimple/CULane/ELAS/LLAMAS, and assumes lane markings are
        (roughly) vertical in image space -- true for forward-facing
        camera setups.
        """
        rows_with_pixels = np.where(mask.any(axis=1))[0]
        if rows_with_pixels.size == 0:
            return None

        points = []
        for row in rows_with_pixels:  # np.where preserves ascending order
            xs = np.nonzero(mask[row])[0]
            x_center = float(xs.mean()) + x_offset
            y_value = float(row) + y_offset
            points.append((x_center, y_value))
        return points

    # ------------------------------------------------------------------
    # Interface expected by PointsDataset (lib/datasets/lane_dataset.py)
    # ------------------------------------------------------------------
    def get_img_heigth(self, path):  # noqa: spelling matches the rest of the codebase
        _, h = self._img_sizes.get(path, (None, None))
        if h is not None:
            return h
        import cv2
        return cv2.imread(path).shape[0]

    def get_img_width(self, path):
        w, _ = self._img_sizes.get(path, (None, None))
        if w is not None:
            return w
        import cv2
        return cv2.imread(path).shape[1]

    def get_metrics(self, lanes, idx):
        # No official benchmark metric is defined for a private COCO
        # dataset; placeholder follows the same convention as elas.py.
        return [1] * len(lanes), [1] * len(lanes), None

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        # Placeholder (same convention as elas.py). Implement a custom
        # metric here if/when you need quantitative test.py results.
        return "", None

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
