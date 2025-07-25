import enum
import logging
import os
from functools import partial
from typing import Union, Tuple, Optional

import torch
import torchvision
from torch import nn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, _default_anchorgen, RetinaNetHead, \
    RetinaNet

try:
    from ScaleNet.pytorch import scalenet
except ModuleNotFoundError:
    logging.info('Scalenet not imported')

try:
    from retinanet_parameters import FPNExtraBlocks, BackboneType
except (ModuleNotFoundError, ImportError):
    from tools.retinanet_parameters import FPNExtraBlocks, BackboneType


def build_retinanet_model(
        score_threshold: float,
        iou_threshold: float,
        image_size: Tuple[int, int],
        max_det: int = 300,
        num_classes: int = 91,
        use_COCO_pretrained_weights: bool = True,
        mean_values: Union[Tuple[float, float, float], None] = None,
        std_values: Union[Tuple[float, float, float], None] = None,
        unfrozen_layers: int = 3,
        trained_weights: Union[str, None] = None,
        anchor_boxes_params: Union[dict, None] = None,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.4,

):
    if trained_weights is None:
        if use_COCO_pretrained_weights:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        else:
            weights = None

    else:
        weights = None

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=weights,
        image_mean=mean_values,
        image_std=std_values,
        score_thresh=score_threshold,
        nms_thresh=iou_threshold,
        detections_per_img=max_det,
        trainable_backbone_layers=unfrozen_layers,
        fg_iou_thresh=fg_iou_thresh,
        bg_iou_thresh=bg_iou_thresh,
        min_size=min(*image_size),
        max_size=max(*image_size)
    )

    if anchor_boxes_params is not None:
        if not isinstance(anchor_boxes_params, dict):
            model.anchor_generator = AnchorGenerator(**{k: v for k, v in anchor_boxes_params.dict().items() if k != 'auto_size'})
        else:
            model.anchor_generator = AnchorGenerator(
                **{k: v for k, v in anchor_boxes_params.items() if k != 'auto_size'})


    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    if trained_weights is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(trained_weights, weights_only=True))
        else:
            model.load_state_dict(torch.load(trained_weights, weights_only=True, map_location=torch.device('cpu')))

    return model


def build_model(
        score_threshold: float,
        use_imageNet_pretrained_weights: bool,
        iou_threshold: float,
        image_size: Tuple[int, int],
        max_det: int = 300,
        num_classes: int = 91,
        mean_values: Union[Tuple[float, float, float], None] = None,
        std_values: Union[Tuple[float, float, float], None] = None,
        unfrozen_layers: int = 3,
        trained_weights: Union[str, None] = None,
        anchor_boxes_params: Union[dict, None] = None,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.4,
        backbone_type: BackboneType = BackboneType.ResNet,
        backbone_layers_nb: int = 50,
        add_P2_to_FPN: bool = False,
        extra_blocks_FPN: Optional[FPNExtraBlocks] = FPNExtraBlocks.LastLevelMaxPool) -> RetinaNet:

    backbone_fpn = build_backbone(backbone_type=backbone_type, size=backbone_layers_nb, add_P2_to_FPN=add_P2_to_FPN,
                                  extra_blocks=extra_blocks_FPN, trainable_backbone_layers=3,
                                  use_imageNet_pretrained_weights=use_imageNet_pretrained_weights)

    if anchor_boxes_params is not None:
        anchor_generator = AnchorGenerator(**{k: v for k, v in anchor_boxes_params.dict().items() if k != 'auto_size'})

    else:
        anchor_generator = _default_anchorgen()

    head = RetinaNetHead(
        backbone_fpn.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"



    model = RetinaNet(backbone=backbone_fpn,
                      num_classes=num_classes,
                      anchor_generator=anchor_generator,
                      head=head,
                      image_mean=mean_values,
                      image_std=std_values,
                      score_thresh=score_threshold,
                      nms_thresh=iou_threshold,
                      detections_per_img=max_det,
                      trainable_backbone_layers=unfrozen_layers,
                      fg_iou_thresh=fg_iou_thresh,
                      bg_iou_thresh=bg_iou_thresh,
                      min_size=min(*image_size),
                      max_size=max(*image_size)
                      )

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    if trained_weights is not None:
        model.load_state_dict(torch.load(trained_weights, weights_only=True))

    return model


def build_scalenet_model(size: int, structures_path: str) -> nn.Module:
    resNet_sizes = [50, 101, 152]
    if size in resNet_sizes:
        kwargs = {
            'structure_path': os.path.join(structures_path, f'scalenet{size}.json'),
        }
        return getattr(scalenet, f"scalenet{size}")(**kwargs)

    else:
        raise ValueError(f"ScaleNet model size {size} is not supported")


def build_resnet_model(size: int, pretrained: bool = False) -> nn.Module:
    resNet_sizes = [18, 34, 50, 101, 152]
    if size in resNet_sizes:
        return getattr(torchvision.models, f'resnet{size}')(pretrained=pretrained)

    else:
        raise ValueError(f"ResNet model size {size} is not supported")

def build_resnext_model(size: int, pretrained: bool = False) -> nn.Module:
    resNeXt_sizes = [50, 101]
    if size in resNeXt_sizes:
        if size == 50:
            return torchvision.models.resnext50_32x4d(pretrained=pretrained)

        else:
            return torchvision.models.resnext101_32x8d(pretrained=pretrained)

    else:
        raise ValueError(f"ResNeXt model size {size} is not supported")


def build_backbone(backbone_type: BackboneType, use_imageNet_pretrained_weights: bool, size: int = 50, add_P2_to_FPN: bool = False,
                   extra_blocks: Optional[FPNExtraBlocks] = FPNExtraBlocks.LastLevelMaxPool,
                   trainable_backbone_layers: int = 3) -> _resnet_fpn_extractor:
    if backbone_type == BackboneType.ScaleNet:
        backbone = build_scalenet_model(size=size, structures_path='ScaleNet/structures')

    elif backbone_type.value == 'ResNeXt':
        backbone = build_resnext_model(size=size, pretrained=use_imageNet_pretrained_weights)

    else:
        backbone = build_resnet_model(size=size, pretrained=use_imageNet_pretrained_weights)

    # is_trained = False
    # trainable_backbone_layers = None
    # trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    if add_P2_to_FPN:
        returned_layers = [2, 3, 4]
    else:
        returned_layers = [1, 2, 3, 4]

    torchvision_extra_block = None
    if extra_blocks is not None:
        if isinstance(extra_blocks, FPNExtraBlocks):
            if extra_blocks == FPNExtraBlocks.LastLevelMaxPool:
                torchvision_extra_block = torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
            elif extra_blocks == FPNExtraBlocks.LastLevelP6P7:
                torchvision_extra_block = torchvision.ops.feature_pyramid_network.LastLevelP6P7(dict(backbone.named_modules())['fc'].in_features, 256)

            else:
                raise ValueError('Invalid extra bloc name: {}'.format(extra_blocks))

        elif isinstance(extra_blocks, enum.Enum):
            if extra_blocks.value == 'LastLevelMaxPool':
                torchvision_extra_block = torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
            elif extra_blocks.value == 'LastLevelP6P7':
                torchvision_extra_block = torchvision.ops.feature_pyramid_network.LastLevelP6P7(
                    dict(backbone.named_modules())['fc'].in_features, 256)

            else:
                raise ValueError('Invalid extra bloc name: {}'.format(extra_blocks))

        # torchvision_extra_block = getattr(torchvision.ops.feature_pyramid_network, extra_blocks.value)()
    else:
        torchvision_extra_block = None

    return _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=returned_layers, extra_blocks=torchvision_extra_block
    )


def collate_fn(batch):
    return tuple(zip(*batch))
