import time
from collections import namedtuple
from collections import namedtuple
from typing import Any, cast, Optional

import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.augmentations.mixing.functional import _preprocess_item_annotations
from albumentations.core.bbox_utils import denormalize_bboxes, clip_bboxes
from albumentations.core.utils import ShapeType
from bboxes import Bbox


class MixedAugmentations(DualTransform):
    def __init__(self, metadata_key: str, target_size: tuple = (512, 512), p: float = 0.5):
        super(MixedAugmentations, self).__init__(p)
        self.target_size = target_size
        self.metadata_key = metadata_key

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates the steps to calculate primary and secondary parameters by calling helper methods."""
        secondary_data = self._get_second_picture_data(data)
        secondary_item = self._preprocess_secondary_item(additional_item=secondary_data, data=data)

        primary = self.get_primary_data(data)

        return {"primary": primary, "secondary": secondary_item}

    @property
    def targets_as_params(self) -> list[str]:
        """Get list of targets that should be passed as parameters to transforms.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    def _get_second_picture_data(self, data: dict):
        return data.get(self.metadata_key)

    def _preprocess_secondary_item(
            self,
            additional_item: dict[str, Any],
            data: dict[str, Any],
    ) -> dict:
        if "bboxes" in data or "keypoints" in data:
            bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
            keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
            return self._preprocess_secondary_annotations(additional_item, bbox_processor, keypoint_processor)

    def _preprocess_secondary_annotations(self, item: dict[str, Any], bbox_processor, keypoint_processor) -> dict[
        str, Optional[Any]]:
        processed_bboxes = _preprocess_item_annotations(item, bbox_processor, "bboxes")
        processed_keypoints = _preprocess_item_annotations(item, keypoint_processor, "keypoints")

        # Construct the final processed item dict
        return {
            "image": item["image"],
            "mask": item.get("mask"),
            "bboxes": processed_bboxes,  # Already np.ndarray or None
            "keypoints": processed_keypoints,  # Already np.ndarray or None
        }

    @staticmethod
    def get_primary_data(data: dict[str, Any]) -> dict[str, Any]:
        """Get a copy of the primary data (data passed in `data` parameter) to avoid modifying the original data.

        Args:
            data (dict[str, Any]): Dictionary containing the primary data.

        Returns:
            fmixing.ProcessedMosaicItem: A copy of the primary data.

        """
        mask = data.get("mask")
        if mask is not None:
            mask = mask.copy()
        bboxes = data.get("bboxes")
        if bboxes is not None:
            bboxes = bboxes.copy()
        keypoints = data.get("keypoints")
        if keypoints is not None:
            keypoints = keypoints.copy()
        return {
            "image": data["image"],
            "mask": mask,
            "bboxes": bboxes,
            "keypoints": keypoints,
        }


class CutMix(MixedAugmentations):
    def __init__(self, bbox_removal_threshold=0.50, min_percentage_cut: float = 0.05, max_percentage_cut: float = 0.5,
                 always_apply=False, metadata_key: str = "mixed_metadata", target_size: tuple = (512, 512),
                 p: float = 0.5):
        super(CutMix, self).__init__(metadata_key, target_size, p)
        self.target_size = target_size
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_percentage_cut = min_percentage_cut
        self.max_percentage_cut = max_percentage_cut
        self.metadata_key = metadata_key

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('bbox_removal_threshold', 'min_percentage_cut', 'max_percentage_cut', 'always_apply', 'metadata_key',
                'target_size', 'p')

    def _select_additional_items(self, data: dict[str, Any], num_additional_needed: int) -> list[dict[str, Any]]:
        from albumentations.augmentations.mixing import functional as fmixing
        valid_items = fmixing.filter_valid_metadata(data.get(self.metadata_key), self.metadata_key, data)
        if len(valid_items) > num_additional_needed:
            return self.py_random.sample(valid_items, num_additional_needed)
        return valid_items

    def _get_cutmix_position(self, img_height, img_width, cutmix_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutmix_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutmix_size + 1),
            np.random.randint(0, img_height - cutmix_size + 1)
        )

    def _get_cutmix(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout size, cutout position)
        """
        cutmix_size = round(np.random.uniform(self.min_percentage_cut, self.max_percentage_cut) * img_width)
        cutmix_position = self._get_cutmix_position(img_height, img_width, cutmix_size)
        return cutmix_size, cutmix_position

    def apply(
            self,
            img: np.ndarray,
            **params: Any,
    ) -> np.ndarray:
        """Apply mosaic transformation to the input image.

        Args:
            img (np.ndarray): Input image
            processed_cells (dict[tuple[int, int, int, int], dict[str, Any]]): Dictionary of processed cell data
            target_shape (tuple[int, int]): Shape of the target image.
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Mosaic transformed image

        """
        image = params['primary']['image'].copy()  # Don't change the original image
        secondary_image = params['secondary']['image'].copy()
        self.img_height, self.img_width, _ = image.shape
        cutout_size, cutout_pos = self._get_cutmix(self.img_height, self.img_width)

        # Set to instance variables to use this later
        # self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size
        self.secondary = params['secondary']

        image[cutout_pos.y:cutout_pos.y + cutout_size, cutout_pos.x:cutout_size + cutout_pos.x, :] = \
            secondary_image[cutout_pos.y:cutout_pos.y + cutout_size, cutout_pos.x:cutout_size + cutout_pos.x, :]
        return cv2.resize(image, self.target_size)

    def filter_bboxes_on_primary_image(self, bboxes) -> np.ndarray:
        return self.filter_bboxes_depending_on_cut_overlap(bboxes=bboxes, filter_inside=True)

    def filter_bboxes_depending_on_cut_overlap(self, bboxes: np.ndarray, filter_inside: bool) -> np.ndarray:
        filtered_bboxes_index = []
        # Denormalize the bbox coordinates
        denorm_bboxes = denormalize_bboxes(bboxes, shape=(self.img_height, self.img_width))

        bbox_patch = Bbox(self.cutout_pos.x, self.cutout_pos.y, self.cutout_pos.x + self.cutout_size,
                          self.cutout_pos.y + self.cutout_size)

        for index, bbox in enumerate(denorm_bboxes):

            x_min, y_min, x_max, y_max, _ = tuple(map(int, bbox))
            bbox = Bbox(x_min, y_min, x_max, y_max)

            overlap_percentage = bbox.intersect(bbox_patch) / bbox.area

            if filter_inside:
                # Remove the bbox if it has more than some threshold of content is inside the cutout patch
                if overlap_percentage <= self.bbox_removal_threshold:
                    filtered_bboxes_index.append(index)
            else:
                if overlap_percentage > self.bbox_removal_threshold:
                    filtered_bboxes_index.append(index)

        return bboxes[filtered_bboxes_index]

    def filter_bboxes_on_secondary_image(self) -> np.ndarray:
        return self.filter_bboxes_depending_on_cut_overlap(bboxes=self.secondary['bboxes'], filter_inside=False)

    def apply_to_bboxes(
            self,
            bboxes: np.ndarray,  # Original bboxes - ignored
            **params: Any,
    ) -> np.ndarray:
        primary_bboxes = self.filter_bboxes_on_primary_image(bboxes=bboxes)
        secondary_bboxes = self.filter_bboxes_on_secondary_image()
        concat_bboxes = np.concatenate((primary_bboxes, secondary_bboxes))
        return clip_bboxes(bboxes=concat_bboxes,
                           shape=ShapeType({'height': self.target_size[0], 'width': self.target_size[1]}))


class MixUp(MixedAugmentations):
    def __init__(self, target_size: tuple[int, int], p: float = 0.5, alpha: float = 0.62,
                 metadata_key: str = "mixed_metadata"):
        super(MixUp, self).__init__(metadata_key, target_size, p)
        self.secondary_bboxes = None
        self.target_size = target_size
        self.metadata_key = metadata_key
        self.alpha = alpha

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return 'alpha', 'metadata_key', 'target_size', 'p'

    def apply(
            self,
            img: np.ndarray,
            **params: Any,
    ) -> np.ndarray:
        """Apply mosaic transformation to the input image.

        Args:
            img (np.ndarray): Input image
            processed_cells (dict[tuple[int, int, int, int], dict[str, Any]]): Dictionary of processed cell data
            target_shape (tuple[int, int]): Shape of the target image.
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Mosaic transformed image

        """
        # keep the boxes of the secondary image to be used in *apply_to_bboxes* function
        self.secondary_bboxes = params['secondary']['bboxes']

        primary_image = params['primary']['image'].copy()  # Don't change the original image
        secondary_image = params['secondary']['image'].copy()

        primary_image = cv2.resize(primary_image, self.target_size)
        secondary_image = cv2.resize(secondary_image, self.target_size)
        image = cv2.addWeighted(primary_image, self.alpha, secondary_image, 1 - self.alpha, 0)
        return image

    def apply_to_bboxes(self, bboxes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        primary_bboxes = clip_bboxes(bboxes=bboxes,
                                     shape=ShapeType({'height': self.target_size[0], 'width': self.target_size[1]}))
        secondary_bboxes = clip_bboxes(bboxes=self.secondary_bboxes,
                                       shape=ShapeType({'height': self.target_size[0], 'width': self.target_size[1]}))
        return np.concatenate((primary_bboxes, secondary_bboxes))
