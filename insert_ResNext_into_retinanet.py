import os

import albumentations as A
import cv2
import torch
import torchvision
from albumentations import ToTensorV2
from torch import nn
from torchvision.models.detection.backbone_utils import _validate_trainable_layers

from ScaleNet.pytorch import scalenet
import sys
sys.path.append('training_image/picsellia_folder')
from training_image.picsellia_folder.dataset import PascalVOCDataset
from tools.model_retinanet import collate_fn, build_model
from tools.retinanet_parameters import TrainingParameters
from training_image.picsellia_folder.utils import read_yaml_file


def build_scalenet_model(size: int, structures_path: str) -> nn.Module:
    resNet_sizes = [50, 102, 152]
    if size in resNet_sizes:
        kwargs = {
            'structure_path': os.path.join(structures_path, f'scalenet{size}.json'),
            'num_classes': num_classes
        }
        return getattr(scalenet, f"scalenet{size}")(**kwargs)

    else:
        raise ValueError(f"This model size {size} is not supported")


def build_resnet_model(size: int, pretrained: bool = False) -> nn.Module:
    resNet_sizes = [18, 34, 50, 101, 152]
    if size in resNet_sizes:
        return getattr(torchvision.models, f'resnet{size}')(pretrained=pretrained)

    else:
        raise ValueError(f"ResNet model size {size} is not supported")





if __name__ == "__main__":
    image_size = (128, 128)
    path_root = r'training_image/datasets'
    single_cls = True
    train_transform = A.Compose([
        A.RandomCrop(*image_size),
        A.Rotate(p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.HueSaturationValue(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    valid_transform = A.Compose([
        A.RandomCrop(*image_size),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    train_dataset = PascalVOCDataset(
        data_folder=os.path.join(path_root, 'train'),
        split='train',
        single_cls=single_cls,
        transform=train_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    # resnet50 = resnet50()

    # retinanet = RetinaNet(backbone=resnet50, num_classes = 2)

    num_classes = 2

    # scalenet = build_scalenet_model(structures_path='ScaleNet/structures',
    #                                 num_classes=num_classes,
    #                                 size=50)

    # resnet = build_resnet_model(size=18, pretrained=False)


    display_progress_bar = True
    weights_backbone = None
    trainable_backbone_layers = None

    is_trained = False
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    # backbone = resnet50(weights=weights_backbone, progress=display_progress_bar)
    # backbone = build_scalenet_model(structures_path='ScaleNet/structures',
    #                                 num_classes=num_classes,
    #                                 size=50)
    # backbone = build_scalenet_model((structures_path='ScaleNet/structures', )

    config = read_yaml_file(file_path=r'training_image/picsellia_folder/config.yaml')
    training_parameters = TrainingParameters(**config)

    # backbone = resnext50_32x4d(pretrained=False)
    # backbone_fpn = _resnet_fpn_extractor(
    #     backbone, trainable_backbone_layers, returned_layers=[1, 2, 3, 4], extra_blocks=LastLevelMaxPool()
    # )
    #
    #
    # # backbone.load_state_dict(torch.load(torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights))
    # # image = torch.zeros((1, 3, 512, 512))
    # # backbone = scalenet_resnet_50(image)
    #
    # # backbone = _resnet_fpn_extractor(
    # #     backbone, trainable_backbone_layers, returned_layers=None, extra_blocks=LastLevelP6P7(2048, 256)
    # # )
    # # backbone_fpn = _resnet_fpn_extractor(
    # #     backbone, trainable_backbone_layers, returned_layers=[1, 2, 3, 4], extra_blocks=LastLevelMaxPool()
    # # )
    # #
    # # print(backbone)
    #
    # anchor_generator = _default_anchorgen()
    # # anchor_generator = AnchorGenerator()
    #
    # head = RetinaNetHead(
    #     backbone_fpn.out_channels,
    #     anchor_generator.num_anchors_per_location()[0],
    #     num_classes,
    #     norm_layer=partial(nn.GroupNorm, 32),
    # )
    # head.regression_head._loss_type = "giou"
    #
    # model = RetinaNet(backbone=backbone_fpn,
    #                   num_classes=num_classes,
    #                   anchor_generator=anchor_generator,
    #                   head=head)

    # model = RetinaNet(backbone=backbone_fpn,
    #                   num_classes=num_classes,
    #                   anchor_generator=anchor_generator,
    #                   head=head,
    #                   image_mean=training_parameters.augmentations.normalization.mean,
    #                   image_std=training_parameters.augmentations.normalization.std,
    #                   score_thresh=training_parameters.confidence_threshold,
    #                   nms_thresh=training_parameters.iou_threshold,
    #                   detections_per_img=1000,
    #                   trainable_backbone_layers=3,
    #                   fg_iou_thresh=training_parameters.fg_iou_thresh,
    #                   bg_iou_thresh=training_parameters.bg_iou_thresh
    #                   )

    model = build_model(num_classes=2,
                        score_threshold=training_parameters.confidence_threshold,
                        iou_threshold=training_parameters.iou_threshold,
                        unfrozen_layers=training_parameters.unfreeze,
                        mean_values=training_parameters.augmentations.normalization.mean,
                        std_values=training_parameters.augmentations.normalization.std,
                        anchor_boxes_params=None,
                        image_size=(1024, 1024),
                        use_imageNet_pretrained_weights=True,
                        # anchor_boxes_params=training_parameters.anchor_boxes,
                        fg_iou_thresh=training_parameters.fg_iou_thresh,
                        bg_iou_thresh=training_parameters.bg_iou_thresh,
                        backbone_type=training_parameters.backbone.backbone_type,
                        backbone_layers_nb=50,
                        add_P2_to_FPN=training_parameters.backbone.add_P2_to_FPN,
                        extra_blocks_FPN=training_parameters.backbone.extra_blocks_FPN
                        )

    print(model)

    model.train()
    image = torch.zeros((1, 3, image_size[0], image_size[1]))
    image = image.to('cuda')
    model.to('cuda')

    targets = next(iter(train_data_loader))[1]
    targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

    results = model(image, targets)
    print(results)
