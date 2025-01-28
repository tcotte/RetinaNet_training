import os
from typing import Union, Annotated, Optional

from pydantic import BaseModel, BeforeValidator


def _transform_str_to_bool(value: Union[bool, str]) -> Union[bool, str]:
    if isinstance(value, str):
        return bool(value.lower())

    return value


def _transform_int_to_tuple(value: Union[int, tuple[int, int]]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value

    return value


BooleanFromString = Annotated[
    bool,
    BeforeValidator(_transform_str_to_bool),
]

TupleFromInt = Annotated[
    tuple[int, int],
    BeforeValidator(_transform_int_to_tuple),
]


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
    mean: list[float, float, float] = [0.485, 0.456, 0.406]
    std: list[float, float, float] = [0.229, 0.224, 0.225]


#
class AugmentationParameters(BaseModel):
    normalization: NormalizationParameters = NormalizationParameters()


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
    image_size: TupleFromInt = (640, 640)
    single_class: BooleanFromString = False
    coco_pretrained_weights: BooleanFromString = False
    weights: Union[None, str] = None
    confidence_threshold: float = 0.2
    iou_threshold: float = 0.5
    unfreeze: int = 3
    patience: int = 50
    loss: LossParameters = LossParameters()


if __name__ == '__main__':
    lr_params = LearningRateParameters(initial_lr=0.2)

    training_parameters = TrainingParameters(
        **{'single_class': 'True',
           'coco_pretrained_weights': 'true',
           'image_size': 1024,
            'loss': {'bbox_regression': 0.7, 'classification': 0.3}}

    )
    print(training_parameters)
