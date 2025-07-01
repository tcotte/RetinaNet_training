import logging

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm



class AnchorBoxOptimizer:
    def __init__(self, dataloader: DataLoader, add_P2_to_FPN: bool):
        self._dataloader = dataloader
        self._add_P2_to_FPN = add_P2_to_FPN
        self._len_anchor_sizes = 5 if not self._add_P2_to_FPN else 6

        self._dataset_bboxes_sizes = self.get_dataset_bboxes_sizes()

        self._labels = self.compute_anchor_boxes_clusters()


    def compute_anchor_boxes_clusters(self):
        logging.info("Computing anchor boxes clusters...")
        returned_layers = [1, 2, 3, 4] if self._add_P2_to_FPN else [2, 3, 4]
        number_anchor_boxes_sizes: int = self._len_anchor_sizes * len(returned_layers)  # same as RetinaNet

        X = np.vstack((np.array(self._dataset_bboxes_sizes['width']),
                       np.array(self._dataset_bboxes_sizes['height']))).T
        K = KMeans(number_anchor_boxes_sizes, random_state=0)
        return K.fit(X)

    def get_dataset_bboxes_sizes(self) -> dict:
        def define_box_shape(boxes: torch.Tensor) -> tuple[np.array, np.array]:
            width_boxes = boxes[:, 2] - boxes[:, 0]
            height_boxes = boxes[:, 3] - boxes[:, 1]
            return width_boxes.numpy(), height_boxes.numpy()

        logging.info('Compute width and height training bounding box sizes')
        dataset_bounding_boxes_sizes: dict = {
            'width': [],
            'height': []
        }

        for _, targets in tqdm(self._dataloader):
            boxes = torch.vstack([i['boxes'] for i in targets])
            size_boxes = define_box_shape(boxes)

            dataset_bounding_boxes_sizes['width'].extend(size_boxes[0])
            dataset_bounding_boxes_sizes['height'].extend(size_boxes[1])

        return dataset_bounding_boxes_sizes

    def get_anchor_boxes_sizes(self) -> tuple:
        returned_layers = [1, 2, 3, 4] if self._add_P2_to_FPN else [2, 3, 4]

        anchor_sizes = self.get_anchor_boxes_sizes_in_RetinaNet_format(
            scales=np.sort(np.mean(self._labels.cluster_centers_, axis=1)),
            returned_layers=returned_layers)

        logging.info("Anchor sizes computed from KMeans: {}".format(anchor_sizes))

        return anchor_sizes

    def plot_boxes_sizes(self):
        return sns.jointplot(x="width", y="height", data=self._dataset_bboxes_sizes)

    def plot_anchor_boxes_clusters(self):
        X = np.vstack((np.array(self._dataset_bboxes_sizes['width']),
                       np.array(self._dataset_bboxes_sizes['height']))).T
        return plt.scatter(X[:, 0], X[:, 1], c=self._labels.labels_, s=50, cmap='viridis')

    def get_anchor_boxes_sizes_in_RetinaNet_format(self, scales: np.array, returned_layers: list) -> np.array:
        list_sizes = np.round(np.reshape(np.sort(scales), (self._len_anchor_sizes, len(returned_layers)))).astype(
            int).tolist()
        return tuple(tuple(x) for x in list_sizes)

#
#
# if __name__ == '__main__':
#     import cv2
#     import matplotlib.pyplot as plt
#
#     image_size = (2048, 2048)
#     add_P2_to_FPN = True
#     path_root = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\datasets'
#     single_cls = True
#     num_workers = 4
#     batch_size = 4
#     train_transform = A.Compose([
#         A.RandomCrop(*image_size),
#         A.Rotate(p=0.5, border_mode=cv2.BORDER_REFLECT),
#         A.HueSaturationValue(p=0.1),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))
#
#     train_dataset = PascalVOCDataset(
#         data_folder=os.path.join(path_root, 'train'),
#         split='train',
#         single_cls=single_cls,
#         transform=train_transform)
#
#
#     train_data_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         num_workers=num_workers,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn
#     )
#
#     dataset_anchor_boxes_sizes = get_dataset_bboxes_sizes(train_data_loader)
#     # sns.jointplot(x="width", y="height", data=dataset_anchor_boxes_sizes)
#
#     returned_layers = [1, 2, 3, 4] if add_P2_to_FPN else [2, 3, 4]
#     number_anchor_boxes_sizes: int = 5*len(returned_layers)  # same as RetinaNet
#
#     X = np.vstack((np.array(dataset_anchor_boxes_sizes['width']), np.array(dataset_anchor_boxes_sizes['height']))).T
#     K = KMeans(number_anchor_boxes_sizes, random_state=0)
#     labels = K.fit(X)
#
#     out = labels.cluster_centers_
#
#     ar = out[:, 0] / out[:, 1]
#     scale = out[:, 1] * np.sqrt(ar) / 256
#
#     print("Aspect Ratios: {}".format(np.sort(ar)))
#
#     print("Scales: {}".format(np.sort(np.mean(out, axis=1))))
#
#     anchor_sizes = get_anchor_boxes_sizes_in_RetinaNet_format(
#         scales=np.sort(np.mean(out, axis=1)),
#         returned_layers=returned_layers)
#     print("Anchor Sizes: {}".format(anchor_sizes))
#     aspect_ratios = _default_anchorgen().aspect_ratios
#     anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=_default_anchorgen().aspect_ratios)
#     print("Anchor Generator cells: {}".format(anchor_generator.cell_anchors))
