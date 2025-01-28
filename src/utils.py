import typing

import torch
import yaml
from torchvision.ops import nms


def get_CUDA_memory_allocation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('No CUDA found')

    else:
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(f"Total memory(MB): {total} /"
              f"Memory used (MB): {mem_used_MB} /"
              f"Memory free (MB): {free}")


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Averager:  ##Return the average loss
    def __init__(self):
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0

    def send(self, new_losses: dict):
        for key, value in self.current_losses.items():
            self.current_losses[key] = new_losses[key] + value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return {
                "regression": 0.0,
                "classification": 0.0,
                "total": 0.0
            }
        else:
            dict_to_return = {}
            for key, value in self.current_losses.items():
                dict_to_return[key] = value / self.iterations

            return dict_to_return

    def reset(self):
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0


def filter_predictions_by_confidence_threshold(predictions: dict, confidence_threshold: float):
    filter_by_confidence = predictions['scores'] > confidence_threshold
    for k, v in predictions.items():
        predictions[k] = predictions[k][filter_by_confidence]
    return predictions


def apply_nms_on_predictions(predictions: dict, iou_threshold: float):
    kept_bboxes = nms(boxes=predictions['boxes'], scores=predictions['scores'], iou_threshold=iou_threshold)

    for k, v in predictions.items():
        if predictions[k].size()[0] != 0:
            predictions[k] = torch.stack([predictions[k][i] for i in kept_bboxes.tolist()])
    # predictions['labels'] = predictions['labels'].type(torch.int64)
    return predictions


def apply_postprocess_on_predictions(predictions: list[dict], iou_threshold: float, confidence_threshold: float):
    post_processed_predictions = []
    for one_picture_prediction in predictions:
        one_picture_prediction = filter_predictions_by_confidence_threshold(predictions=one_picture_prediction,
                                                                            confidence_threshold=confidence_threshold)
        one_picture_prediction = apply_nms_on_predictions(predictions=one_picture_prediction,
                                                          iou_threshold=iou_threshold)
        post_processed_predictions.append(one_picture_prediction)
    return predictions


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def apply_loss_weights(loss_dict: dict, loss_coefficients: dict) -> dict:
    for k in loss_dict.keys():
        loss_dict[k] *= loss_coefficients[k]
    return loss_dict


def read_configuration_file(config_file: str) -> typing.Union[dict, None]:
    with open(config_file) as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return configs


def get_GPU_occupancy(gpu_id: int = 0) -> float:
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info(device=gpu_id)
        return 1 - free_memory / total_memory

    else:
        return 0.0
