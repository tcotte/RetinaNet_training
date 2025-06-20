import os
from enum import Enum
from typing import Union, Optional, Tuple, List, Any
from distutils.util import strtobool
from pydantic import BaseModel, field_validator


class BackboneType(Enum):
    ResNet = 'ResNet'
    ScaleNet = 'ScaleNet'


class FPNExtraBlocks(Enum):
    LastLevelMaxPool = 'LastLevelMaxPool'
    LastLevelP6P7 = 'LastLevelP6P7'


class LRWarmupParameters(BaseModel):
    last_step: int = 5
    warmup_period: int = 2


class LRDecayParameters(BaseModel):
    step_size: int = 50
    gamma: float = 0.5


class LearningRateParameters(BaseModel):
    initial_lr: float = 0.001
    warmup: LRWarmupParameters = LRWarmupParameters()
    decay: LRDecayParameters = LRDecayParameters()


class LossParameters(BaseModel):
    bbox_regression: float = 0.5
    classification: float = 0.5


class NormalizationParameters(BaseModel):
    mean: List[float] = [0.485, 0.456, 0.406]
    std: List[float] = [0.229, 0.224, 0.225]
    auto_norm: bool = False

    @field_validator("auto_norm", mode='before')
    def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
        if isinstance(value, str):
            return bool(strtobool(value))

        return value


#
class AugmentationParameters(BaseModel):
    version: int = 1
    crop: bool = False
    normalization: NormalizationParameters = NormalizationParameters()

    @field_validator("crop", mode='before')
    def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
        if isinstance(value, str):
            return bool(strtobool(value))

        return value


class AnchorBoxesParameters(BaseModel):
    sizes: Optional[tuple[
        tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int],
        tuple[int, int, int]]] = None
    aspect_ratios: Optional[tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float],
        tuple[float, float, float]]] = None
    auto_size: Optional[bool] = False

    @field_validator("auto_size", mode='before')
    def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
        if isinstance(value, str):
            return bool(strtobool(value))

        return value

    @field_validator("sizes", "aspect_ratios", mode="before")
    def _transform_str_to_tuple(value: Union[str, tuple]) -> tuple:
        if isinstance(value, str):
            return eval(value)

        return value


class BackboneParameters(BaseModel):
    backbone_type: BackboneType = BackboneType.ResNet
    backbone_layers_nb: int = 50
    add_P2_to_FPN: bool = False
    extra_blocks_FPN: FPNExtraBlocks = FPNExtraBlocks.LastLevelP6P7

    @field_validator('add_P2_to_FPN', mode='before')
    def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
        if isinstance(value, str):
            return bool(strtobool(value))

        return value


class TrainingParameters(BaseModel):
    augmentations: AugmentationParameters = AugmentationParameters()
    epoch: int = 100
    batch_size: int = 8
    device: str = 'cpu'
    device_name: Optional[str] = None
    learning_rate: LearningRateParameters = LearningRateParameters()
    weight_decay: float = 0.0005
    optimizer: str = 'Adam'
    workers_number: int = os.cpu_count()
    image_size: Tuple[int, int] = (640, 640)
    single_class: bool = False
    coco_pretrained_weights: bool = False
    weights: Union[None, str] = None
    confidence_threshold: float = 0.2
    iou_threshold: float = 0.5
    unfreeze: int = 3
    patience: int = 50
    loss: LossParameters = LossParameters()
    anchor_boxes: Optional[AnchorBoxesParameters] = None
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4
    backbone: BackboneParameters = BackboneParameters()

    # def transform_id_to_str(cls, value) -> str:
    #     return str(value)

    @field_validator("single_class", "coco_pretrained_weights", mode='before')
    def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
        if isinstance(value, str):
            return bool(strtobool(value))

        return value

    @field_validator("image_size", mode='before')
    def _transform_int_to_tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(value, int):
            return value, value

        return value


if __name__ == '__main__':
    lr_params = LearningRateParameters(initial_lr=0.2)

    training_parameters = TrainingParameters(
        **{'single_class': 'True',
           'coco_pretrained_weights': 'true',
           # 'image_size': 2,
           'loss': {'bbox_regression': 0.7, 'classification': 0.3},
           'backbone': {'backbone_type': 'ScaleNet'},
           'fg_iou_thresh': 0.1,
           'anchor_boxes': {
               'sizes': ((32, 40, 50),
                         (64, 80, 101),
                         (128, 161, 203),
                         (256, 322, 406),
                         (512, 645, 812)),
               'aspect_ratios': ((0.5, 1.0, 2.0),
                                 (0.5, 1.0, 2.0),
                                 (0.5, 1.0, 2.0),
                                 (0.5, 1.0, 2.0),
                                 (0.5, 1.0, 2.0)),
               'auto_size': 'True'
           },
           }

    )

    print(training_parameters.image_size)
    print(training_parameters.anchor_boxes)

    training_parameters.anchor_boxes.sizes = ((8, 11, 15, 21), (30, 45, 67, 92), (124, 171, 227, 303), (388, 507, 558, 588), (709, 1005, 1298, 1982))
    print(training_parameters.anchor_boxes)
    print(training_parameters.fg_iou_thresh)



    # torchvision.ops.feature_pyramid_network
    # import ExtraFPNBlock
