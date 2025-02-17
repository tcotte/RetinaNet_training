import os
from enum import Enum, IntEnum
from functools import partial
from typing import Optional

import torch
import torchvision
import yaml
from torch import nn
from torchvision.models.detection import RetinaNet
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.retinanet import _default_anchorgen, RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7, LastLevelMaxPool

from ScaleNet.pytorch import scalenet
from training_image.picsellia_folder.retinanet_parameters import TrainingParameters


def build_scalenet_model(size: int, num_classes: int, structures_path: str) -> nn.Module:
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


def read_config_file(config_file_path: str) -> Optional[dict]:
    with open(config_file_path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


if __name__ == "__main__":
    # resnet50 = resnet50()

    # retinanet = RetinaNet(backbone=resnet50, num_classes = 2)

    num_classes = 2

    scalenet = build_scalenet_model(structures_path='ScaleNet/structures',
                                    num_classes=num_classes,
                                    size=50)

    resnet = build_resnet_model(size=18, pretrained=False)

    display_progress_bar = True
    weights_backbone = None
    trainable_backbone_layers = None

    is_trained = False
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    # backbone = resnet50(weights=weights_backbone, progress=display_progress_bar)
    backbone = scalenet

    config = read_config_file(config_file_path=r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\config.yaml')
    training_parameters = TrainingParameters(**config)

    # backbone.load_state_dict(torch.load(torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights))
    # image = torch.zeros((1, 3, 512, 512))
    # backbone = scalenet_resnet_50(image)

    # backbone = _resnet_fpn_extractor(
    #     backbone, trainable_backbone_layers, returned_layers=None, extra_blocks=LastLevelP6P7(2048, 256)
    # )
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=None, extra_blocks=LastLevelMaxPool()
    )

    print(backbone)

    # anchor_generator = _default_anchorgen()
    anchor_generator = AnchorGenerator(**training_parameters.anchor_boxes.dict())

    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"
    model = RetinaNet(backbone, num_classes, anchor_generator=anchor_generator, head=head)

    model.eval()
    image = torch.zeros((1, 3, 512, 512))
    image = image.to('cuda')
    model.to('cuda')
    results = model(image)
    print(results)
