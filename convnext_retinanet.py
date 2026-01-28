import time

import timm
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator


# class ConvNeXtBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.body = timm.create_model(
#             "convnext_tiny",
#             features_only=True,
#             pretrained=True
#         )
#         self.out_channels = [96, 192, 384, 768]

from torchvision.models.feature_extraction import create_feature_extractor

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import BackboneWithFPN


convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
# Freeze stages 0–2
for name, param in convnext.named_parameters():
    if any([name.startswith(f"features.{i}") for i in [0,1,2]]):
        param.requires_grad = False

# print(convnext.parameters())
return_layers = {
    "1": "c2",
    "3": "c3",
    "5": "c4",
    "7": "c5",
}

backbone = BackboneWithFPN(
    convnext.features,
    return_layers=return_layers,
    in_channels_list=[96, 192, 384, 768],
    out_channels=256,
)
print(backbone.out_channels)

anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32),) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

# backbone = ConvNeXtBackbone()
num_classes = 2

model = RetinaNet(
    backbone=backbone,
    num_classes=num_classes,
    anchor_generator=anchor_generator
)


if __name__ == "__main__":


    model.cuda()

    model.eval()

    x = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    x = x.to('cuda')

    for i in range(3):
        start = time.time()
        out = model(x)
        print(time.time() - start)

