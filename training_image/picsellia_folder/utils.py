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


class Averager:
    """
    Computes and stores the average object detection losses. Object detection losses are: *regression* for bounding box
    regression and classification for object identification. Total is the sum of all losses with coefficients applied
    for each loss type: total = aplha*regression + beta*classification.

    The aim of this class is to sum all losses in each batch of an epoch and be able to compute the average of all
    losses at the end of the epoch.
    """
    def __init__(self):
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0

    def send(self, new_losses: typing.Dict) -> None:
        """
        Increment the iteration attribute and store losses for each loss type: each loss type is summed over all
        iterations.
        :param new_losses: dictionary with all losses types as keys and loss values as values.
                            {"regression":0.2, "classification":0.2, "total":0.2}
        """
        for key, value in self.current_losses.items():
            self.current_losses[key] = new_losses[key] + value
        self.iterations += 1

    @property
    def value(self) -> typing.Dict:
        """
        Get values of current losses.
        :return: dictionary with all losses types as keys and current summed values as values.
        """
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

    def reset(self) -> None:
        """
        Reset losses values and iteration attribute. This function has to be called at the end of each epoch.
        """
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0


def filter_predictions_by_confidence_threshold(predictions: typing.Dict, confidence_threshold: float):
    filter_by_confidence = predictions['scores'] > confidence_threshold
    for k, v in predictions.items():
        predictions[k] = predictions[k][filter_by_confidence]
    return predictions


def apply_nms_on_predictions(predictions: typing.Dict, iou_threshold: float):
    kept_bboxes = nms(boxes=predictions['boxes'], scores=predictions['scores'], iou_threshold=iou_threshold)

    for k, v in predictions.items():
        if predictions[k].size()[0] != 0:
            predictions[k] = torch.stack([predictions[k][i] for i in kept_bboxes.tolist()])
    # predictions['labels'] = predictions['labels'].type(torch.int64)
    return predictions


def apply_postprocess_on_predictions(predictions: typing.List[typing.Dict], iou_threshold: float,
                                     confidence_threshold: float):
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
        """
        The aim of this class is to terminate the training earlier than expected if the model is not improving by a
        certain number of epochs.
        :param patience: number of epochs to wait before stopping if the model does not improve.
        :param min_delta: accepted delta between the current validation loss and the EPOCH-1 validation loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """
        Function called when validation loss is computed to compare the current validation loss and the EPOCH-1
        validation loss. If the validation loss is not improve during *patience* epochs, the training is stopped.
        :param validation_loss: current validation loss value.
        :return: boolean which indicates if we have to stop the training or not.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def apply_loss_weights(loss_dict: typing.Dict, loss_coefficients: typing.Dict) -> typing.Dict:
    """
    Apply coefficient on losses. Considering loss dict keys as 'regression' / 'classification' and loss coefficients as
    alpha / beta -> aplha*regression / beta*classification.
    :param loss_dict: loss dictionary with keys as loss types and values as loss values for each type of loss.
    :param loss_coefficients: coefficient applied for each loss type. Dictionary with keys as loss types and values as
    coefficients for each type of loss.
    :return: dictionary with keys as loss types and values as weighted loss values.
    """
    for k in loss_dict.keys():
        loss_dict[k] *= loss_coefficients[k]
    return loss_dict


def read_configuration_file(config_file: str) -> typing.Union[typing.Dict, None]:
    """
    Read configuration yaml file.
    :param config_file: path of the configuration file.
    :return: dictionary which sums up information in the configuration file.
    """
    with open(config_file) as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return configs


def get_GPU_occupancy(gpu_id: int = 0) -> float:
    """
    Get memory occupancy of the used GPU for training.
    :param gpu_id: id of the GPU used for training model.
    :return: Memory occupancy in percentage.
    """
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info(device=gpu_id)
        return 1 - free_memory / total_memory

    else:
        return 0.0
