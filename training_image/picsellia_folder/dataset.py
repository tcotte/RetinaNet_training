import logging
import os
import random
import xml.etree.ElementTree as ET
from typing import Union

import albumentations
import cv2
import imutils
import imutils.paths
import numpy as np
import torch
from PIL import Image
from albumentations import ToTensorV2
from albumentations.augmentations.mixing.functional import _preprocess_item_annotations
from albumentations.core.bbox_utils import denormalize_bboxes
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.model_retinanet import collate_fn
from advanced_augmentations.cut_out import CutOut
from advanced_augmentations.mixed_augmentations import MixUp, CutMix


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, single_cls, do_class_mapping=True, add_bckd_as_class: bool = True,
                 transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard annotation_files that are considered difficult to detect?
        """
        self.split = split.upper()
        self._single_cls = single_cls
        self._add_bckd_as_class = add_bckd_as_class

        self.transform = transform

        self._apply_mosaic = self._processing_in_transform(
            processing=albumentations.augmentations.mixing.transforms.Mosaic)
        self._apply_cutmix = self._processing_in_transform(processing=CutMix)
        self._apply_mixup = self._processing_in_transform(processing=MixUp)

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        self.images = list(imutils.paths.list_images(os.path.join(self.data_folder, 'JPEGImages')))

        annotation_folder = os.path.join(self.data_folder, 'Annotations')
        self.annotation_files = list([os.path.join(annotation_folder, i) for i in os.listdir(annotation_folder)])

        assert len(self.images) == len(self.annotation_files)

        if do_class_mapping:
            self.class_mapping, self.number_obj_by_cls = self.get_class_mapping()

    def _processing_in_transform(self, processing: Union['CutMix', 'MixUp', albumentations.DualTransform]):
        for trans in self.transform.transforms:
            if type(trans) == processing:
                return True

            elif type(trans) == albumentations.core.composition.OneOf:
                for x in trans.transforms:
                    if type(x) == processing:
                        return True

        return False

    def get_class_mapping(self):
        logging.info(f'Compute class mapping for {self.split} dataset')

        list_classes = []
        dict_classes = {}

        for xml_file in tqdm(self.annotation_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in list_classes:
                    list_classes.append(cls)

                if cls not in dict_classes.keys():
                    dict_classes[cls] = 1
                else:
                    dict_classes[cls] += 1

        class_mapping = {}
        for idx, value in enumerate(list_classes):
            class_mapping[idx] = value

        if self._single_cls:
            if self._add_bckd_as_class:
                class_mapping = {0: 'background', 1: 'cls0'}
            else:
                class_mapping = {0: 'cls0'}
            dict_classes = {'cls0': sum(dict_classes.values())}

        return class_mapping, dict_classes

    def parse_annotation(self, xml_file, single_cls: bool = False):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find("filename").text
        image_path = os.path.join(self.data_folder, 'JPEGImages', image_name)

        boxes = []
        classes = []
        for obj in root.iter("object"):
            cls = obj.find("name").text
            classes.append(cls)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            if xmin >= xmax:
                # print(f"Error loading bounding box in {image_name}")
                pass
            elif ymin >= ymax:
                pass
                # print(f"Error loading bounding box in {image_name}")
            else:
                boxes.append([xmin, ymin, xmax, ymax])

        if not single_cls:
            class_ids = [
                list(self.class_mapping.keys())[list(self.class_mapping.values()).index(cls)]
                for cls in classes
            ]
        else:
            if self._add_bckd_as_class:
                class_ids = [1] * len(boxes)
            else:
                class_ids = [0] * len(boxes)

        boxes, class_ids = self.remove_duplicate_labels(boxes=boxes, labels=class_ids)

        return {'boxes': boxes, 'labels': class_ids, 'image': image_path}

    @staticmethod
    def remove_duplicate_labels(boxes, labels):
        boxes_labels_stack = np.hstack((np.array(boxes), np.expand_dims(np.array(labels), axis=1)))
        boxes_labels_stack_without_duplicate = np.unique(boxes_labels_stack, axis=0)
        return boxes_labels_stack_without_duplicate[:, :4], boxes_labels_stack_without_duplicate[:, -1]

    def __getitem__(self, i):
        # Read annotation_files in this image (bounding boxes, labels, difficulties)
        objects = self.parse_annotation(xml_file=self.annotation_files[i], single_cls=True)

        # Read image
        image = Image.open(objects['image'], mode='r')
        image = image.convert('RGB')

        # difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        augmentations_metadata = {}

        if self._apply_mosaic:
            # Prepare mosaic_metadata: 3 additional samples (total of 4 for mosaic)
            mosaic_candidates = list(range(0, i)) + list(range(i + 1, len(self.annotation_files)))
            mosaic_indices = random.sample(mosaic_candidates, k=3)

            mosaic_metadata = []
            for idx in mosaic_indices:
                ann = self.parse_annotation(self.annotation_files[idx], single_cls=True)
                mosaic_image = Image.open(ann['image']).convert('RGB')
                mosaic_np = np.array(mosaic_image)
                mosaic_metadata.append({
                    "image": mosaic_np,
                    "bboxes": ann["boxes"],
                    "class_labels": ann["labels"]
                })

            augmentations_metadata['mosaic_metadata'] = mosaic_metadata

        if self._apply_cutmix or self._apply_mixup:
            # Prepare cutmix_metadata: 1 additional sample
            mixed_candidates = list(range(0, i)) + list(range(i + 1, len(self.annotation_files)))
            mixed_index = random.sample(mixed_candidates, k=1)[0]

            ann = self.parse_annotation(self.annotation_files[mixed_index], single_cls=True)
            mixed_image = Image.open(ann['image']).convert('RGB')
            mixed_np = np.array(mixed_image)
            mixed_metadata = {
                "image": mixed_np,
                "bboxes": ann["boxes"],
                "class_labels": ann["labels"]
            }

            augmentations_metadata['mixed_metadata'] = mixed_metadata

        if augmentations_metadata == {}:
            transformed = self.transform(image=np.array(image),
                                         bboxes=np.array(objects['boxes']),
                                         class_labels=np.array(objects['labels']))
        else:
            transformed = self.transform(image=np.array(image),
                                         bboxes=np.array(objects['boxes']),
                                         class_labels=np.array(objects['labels']),
                                         **augmentations_metadata)

        image = transformed['image'] / 225.
        boxes = torch.FloatTensor(transformed['bboxes'])  # (n_objects, 4)
        labels = torch.LongTensor(transformed['class_labels'])  # (n_objects)

        if not self.split == 'TEST':
            return image, {'boxes': boxes, 'labels': labels}

        else:
            return image, {'boxes': boxes, 'labels': labels}, os.path.basename(objects['image'])


    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of annotation_files, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels


class PascalVOCTestDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, image_folder, transform=None):
        """
        :param image_folder: folder where images are stored
        """
        self.images = list(imutils.paths.list_images(image_folder))
        self.transform = transform

    def __getitem__(self, i):
        # Read image
        file_path = self.images[i]
        image = Image.open(file_path, mode='r')
        image = image.convert('RGB')

        # Apply transformations
        transformed = self.transform(image=np.array(image))

        image = transformed['image'] / 225.

        return image, os.path.basename(file_path)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import albumentations as A

    DATA_VALIDATION_DIR = r'C:\Users\tristan_cotte\PycharmProjects\datasets\train'
    IMAGE_SIZE = (1024, 1024)
    SINGLE_CLS = True

    train_transform = A.Compose([
        A.OneOf([
            # Cutout(p=0.5),
            # A.Resize(height=1024, width=1024, p=1),
            MixUp(p=1, target_size=(1024, 1024)),
            CutMix(p=1, target_size=(1024, 1024)),
            # CutOut(p=0.3, target_size=(1024, 1024)),
            # A.Mosaic(p=1, target_size=(1024, 1024))

            # MixUp(p=0.25, target_size=(1024, 1024)),
        ], p=1),
        # Cutout(p=1, min_cutout_size=1024, max_cutout_size=2048),

        # A.Normalize(),
        # A.Resize(*IMAGE_SIZE),
        A.VerticalFlip(p=0.5),
        # CustomCutout(p=1, always_apply=True, min_cutout_size=512),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    #
    #     train_transform = A.Compose([
    #         # A.Normalize(mean=[0.9629258011853685, 1.1043921727662964, 0.9835339608076883],
    #         #             std=[0.08148765554920795, 0.10545005065566, 0.13757230267160245],
    #         #             max_pixel_value= 207),
    #         A.RandomCrop(*IMAGE_SIZE),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
    #         A.OneOf([
    #             A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1, p=0.5),
    #             A.GridDropout(
    #                 ratio=0.1,
    #                 unit_size_range=None,
    #                 random_offset=True,
    #                 p=0.2),
    #         ]),
    #         A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.2, 0.2), p=0.5),
    #         A.OneOf([
    #             A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    #             A.HueSaturationValue(p=0.1),
    #             A.RandomBrightnessContrast(p=0.2),
    #         ]),
    #         A.OneOf([
    #             A.ElasticTransform(alpha=1, sigma=50, interpolation=1, approximate=False, same_dxdy=False,
    #                                mask_interpolation=0, noise_distribution='gaussian', p=0.2),
    #             A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, normalized=True,
    #                              mask_interpolation=0, p=0.3)
    #         ]),
    #         A.RandomScale(scale_limit=(-0.3, 0.3), interpolation=1, mask_interpolation=0, p=0.5),
    #
    #         ToTensorV2()
    #     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))
    #
    val_dataset = PascalVOCDataset(
        data_folder=DATA_VALIDATION_DIR,
        split='test',
        single_cls=SINGLE_CLS,
        transform=train_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    # for i in val_dataset[:8]:
    # print(val_dataset[0][0])
    for e in tqdm(range(10000)):
        for torch_image, label, _ in val_loader:
            print(e)
            # # sample = next(iter(val_loader))
            # annotations = label[0]
            # image = torch.permute(torch_image[0], (1, 2, 0)).numpy()
            # image = np.ascontiguousarray(image, dtype=np.float32)
            # for bbox in annotations['boxes']:
            #     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # plt.imshow(image)
            # plt.show()
