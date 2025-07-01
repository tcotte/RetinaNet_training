import albumentations as A
import cv2
from albumentations import ToTensorV2
from matplotlib import pyplot as plt

import yaml

from training_image.picsellia_folder.utils import read_yaml_file


def train_augmentation_v1(random_crop, image_size: tuple[int, int]) -> A.Compose:
    return A.Compose([
        A.RandomCrop(*image_size) if random_crop else A.Resize(*image_size),
        A.Rotate(p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.HueSaturationValue(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))


# Define a more aggressive augmentation pipeline for object detection
def train_augmentation_v2(random_crop, image_size: tuple[int, int]) -> A.Compose:
    return A.Compose([
        # Horizontal flip with high probability
        A.HorizontalFlip(p=0.5),
        A.Affine(p=1, scale=0.8, shear=5, translate_percent=0.05, rotate=15),
        A.PlanckianJitter(p=0.5),

        # Vertical flip with high probability
        A.VerticalFlip(p=0.5),

        # Random rotation (up to 90 degrees)
        A.Rotate(limit=90, p=0.7, border_mode=cv2.BORDER_CONSTANT),

        # Random scale (stretch image)
        A.RandomScale(scale_limit=0.5, p=0.5),

        # Translate image (random shift along x/y axis)
        A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.RandomCrop(*image_size) if random_crop else A.Resize(*image_size),

        # Random brightness and contrast (more aggressive)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

        # Adjust hue, saturation, and value (color jitter)
        A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=0.7),

        # Random perspective distortion (to simulate varying angles)
        A.Perspective(scale=(0.05, 0.1), p=0.7),

        # Gaussian noise (more aggressive noise injection)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

        # Elastic deformation for more complex distortions
        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),

        # Apply blur
        A.GaussianBlur(blur_limit=(5, 15), p=0.3),

        A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=0.2),

        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))


def train_augmentation_v3(random_crop, image_size: tuple[int, int], **kwargs) -> A.Compose:
    """
    The aim of this function will be to implement a similar augmentations as YOLOv8 can do:
    https://docs.ultralytics.com/guides/yolo-data-augmentation/
    :param random_crop:
    :param image_size:
    :return:
    """

    parameters = {
        'HueSaturationValue': {
            'hue': 0.015,
            'sat': 0.7,
            'val': 0.4,
            'prob': 1.0
        },
        'Rotate': {
            'prob': 0.0,
            'limit': 90
        },
        'Translation': {
            'shift_limit': 0.5,
            'prob': 0.1
        },
        'Scale': {
            'scale_limit': 0.5,
            'prob': 0.5
        },
        'VerticalFlip': {
            'prob': 0.0
        },
        'HorizontalFlip': {
            'prob': 0.5
        },
        'Mosaic': {
            'prob': 0.5
        }
    }

    parameters.update(kwargs)

    return A.Compose([

        # Color Space Augmentations
        A.HueSaturationValue(
            hue_shift_limit=parameters['HueSaturationValue']['hue'] * 255,
            sat_shift_limit=parameters['HueSaturationValue']['sat'] * 255,
            val_shift_limit=parameters['HueSaturationValue']['val'] * 255,
            p=parameters['HueSaturationValue']['prob']),

        # Geometric transformations
        A.Rotate(p=parameters['Rotate']['prob'], limit=(-parameters['Rotate']['limit'], parameters['Rotate']['limit']),
                 border_mode=cv2.BORDER_CONSTANT),

        # Translation
        A.ShiftScaleRotate(
            shift_limit=(parameters['Translation']['shift_limit'], parameters['Translation']['shift_limit']),
            scale_limit=(0, 0),
            rotate_limit=(0, 0),
            p=parameters['Translation']['prob']),
        # Scale
        A.ShiftScaleRotate(shift_limit=(0, 0),
                           scale_limit=(-parameters['Scale']['scale_limit'], parameters['Scale']['scale_limit']),
                           rotate_limit=(0, 0),
                           p=parameters['Scale']['prob']),
        # shear -> not implemented -> p =0
        # Perspective -> not implemented -> p =0
        # Flip Up-Down
        A.VerticalFlip(p=parameters['VerticalFlip']['prob']),
        # Flip Left - Right
        A.HorizontalFlip(p=parameters['HorizontalFlip']['prob']),

        # BGR Channel Swap -> not implemented -> p =0

        # Mosaic
        A.OneOf([
            A.Resize(*image_size),
            A.Mosaic(
                grid_yx=[2, 2],
                target_size=image_size,
                cell_shape=image_size,
                center_range=[0.5, 0.5],
                fit_mode="cover",
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=parameters['Mosaic']['prob']
            )
        ]),

        # Mixup(mixup) -> not implemented -> p =0
        # CutMix (cutmix) -> not implemented -> p =0

        # Random Erasing (erasing) -> seems like it is useful only for classification

        # A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))


if __name__ == '__main__':
    import numpy as np

    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White


    def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img


    def visualize(image, bboxes, category_ids, category_id_to_name):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.imshow(img)


    image = cv2.imread(r"C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\dataset\test\JPEGImages\2025_02_13-11_05_29.jpg", cv2.IMREAD_COLOR_RGB)



    d = read_yaml_file(file_path=r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\training_image\picsellia_folder\aug_hyp.yaml')

    for i in range(10):
        transform = train_augmentation_v3(random_crop=False, image_size=(1024, 1024), **d)
        transformed = transform(image=image, class_labels=['d'], bboxes=[], category_ids=[])
        visualize(
            np.transpose(transformed['image'].numpy(), (1, 2, 0)),
            transformed["bboxes"],
            transformed["category_ids"],
            {},
        )
        print(transformed['image'].shape)
        plt.show()
