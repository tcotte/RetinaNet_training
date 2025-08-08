from collections import namedtuple
from typing import Optional

import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.core.bbox_utils import denormalize_bboxes, clip_bboxes
from albumentations.core.utils import ShapeType


class CutOut(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes
    Note: (only supports square cutout regions)

    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """

    def __init__(
            self,
            fill_value=0,
            bbox_removal_threshold=0.50,
            min_percentage_cut=0.05,
            max_percentage_cut=0.5,
            target_size: Optional[tuple] = None,
            always_apply=False,
            p=0.5
    ):
        """
        Class construstor

        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_percentage_cut: minimum percentage of image which will be cut out
        :param max_percentage_cut: maximum percentage of image which will be cut out
        """
        super(CutOut, self).__init__(p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_percentage_cut = min_percentage_cut
        self.max_percentage_cut = max_percentage_cut
        self.target_size = target_size

    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )

    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = round(np.random.uniform(self.min_percentage_cut, self.max_percentage_cut) * img_width)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position

    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image

        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)

        # Set to instance variables to use this later
        self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size

        image[cutout_pos.y:cutout_pos.y + cutout_size, cutout_pos.x:cutout_size + cutout_pos.x, :] = cutout_arr
        if self.target_size is None:
            return image

        else:
            return cv2.resize(image, self.target_size)

    def apply_to_bboxes(self, original_bboxes: np.ndarray, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout
        :param original_bboxes: 2D array of bounding boxes coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """
        filtered_bboxes_index = []
        # Denormalize the bbox coordinates
        bboxes = denormalize_bboxes(original_bboxes, shape=(self.img_height, self.img_width))
        for index, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max, _ = tuple(map(int, bbox))

            bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
            overlapping_size = np.sum(
                (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
                (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
                (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
            )

            # Remove the bbox if it has more than some threshold of content is inside the cutout patch
            if not overlapping_size / bbox_size > self.bbox_removal_threshold:
                filtered_bboxes_index.append(index)
                # return normalize_bboxes(np.array([0, 0, 0, 0]), shape=(self.img_height, self.img_width))

        if self.target_size is None:
            return original_bboxes[filtered_bboxes_index]

        else:
            return clip_bboxes(bboxes=original_bboxes,
                               shape=ShapeType({'height': self.target_size[0], 'width': self.target_size[1]}))

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return 'fill_value', 'bbox_removal_threshold', 'min_percentage_cut', 'max_percentage_cut', 'always_apply', 'p'
