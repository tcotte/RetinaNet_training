from typing import List, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from picsellia import Experiment
from picsellia.types.enums import InferenceType
from torchvision.models.detection import RetinaNet


def plot_precision_recall_curve(validation_metrics: Dict, recall_thresholds: List[float]) -> plt.plot:
    """
    Plot precision-recall curve depending on validation metrics.
    :param validation_metrics: validation COCO metrics at the end of the training
    :param recall_thresholds: recall thresholds used to compute the validation metric
    :return: plot of the precision-recall curve
    """
    recall_values = np.array(recall_thresholds)
    precision_values = np.array(
        [validation_metrics['precision'][0][i][0][0][-1] for i in range(len(recall_thresholds))])

    f1_scores = 2 * (precision_values * recall_values) / (precision_values + recall_values)
    best_threshold_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_threshold_index]

    fig, ax = plt.subplots()

    ax.plot(precision_values, recall_values, label=f'best F1: {best_f1:.2f}')

    ax.axhline(y=precision_values[best_threshold_index], color="red", linestyle='--', lw=1)
    ax.axvline(x=recall_values[best_threshold_index], color="red", linestyle='--', lw=1)

    yticks = [*ax.get_yticks(), precision_values[best_threshold_index]]
    yticklabels = [*ax.get_yticklabels(), float(round(precision_values[best_threshold_index], 2))]
    ax.set_yticks(yticks, labels=yticklabels)

    xticks = [*ax.get_xticks(), recall_values[best_threshold_index]]
    xticklabels = [*ax.get_xticklabels(), float(round(recall_values[best_threshold_index], 2))]
    ax.set_xticks(xticks, labels=xticklabels)

    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(loc='lower left')

    return ax


def fill_picsellia_evaluation_tab(model: RetinaNet, data_loader, experiment: Experiment,
                                  dataset_version_name: str, device, batch_size: int = 1) -> None:
    """
    Fill picsellia evaluation tab to have a visual representation of the inferences on the dataset split sent via
    loader.
    WARNING: This function only works with a batch size of 1.
    TODO: Enables this function to work with a bigger batch size
    :param model: model used to do the predictions
    :param data_loader: dataloader which gathers the data to predict on
    :param experiment: Picsellia experiment where the predictions will be saved
    :param dataset_version_name: name of the dataset version on which the predictions will be done
    :param device: device which will compute the predictions
    :param batch_size: size of the batch to use for predictions
    """
    dataset_version = experiment.get_dataset(name=dataset_version_name)
    picsellia_labels = dataset_version.list_labels()

    model.topk_candidates = 5000
    model.detections_per_img = 3000

    model.to(device)
    model.eval()

    for i, (images, file_paths) in enumerate(data_loader):
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            predictions = model(images)

        for idx in range(batch_size):
            asset = dataset_version.find_asset(filename=file_paths[idx])

            resized_image_height, resized_image_width = images[idx].size()[-2:]
            original_height = asset.height
            original_width = asset.width

            # calculate the scale factors for width and height
            width_scale = original_width / resized_image_width
            height_scale = original_height / resized_image_height

            picsellia_rectangles: list = []
            for box, label, score in zip(predictions[idx]['boxes'], predictions[idx]['labels'],
                                         predictions[idx]['scores']):
                box = box.cpu().numpy()
                label = int(np.squeeze(label.cpu().numpy()))
                score = float(np.squeeze(score.cpu().numpy()))
                rectangle = (int(round(box[0] * width_scale)),
                             int(round(box[1] * height_scale)),
                             int(round((box[2] - box[0]) * width_scale)),
                             int(round((box[3] - box[1]) * height_scale)),
                             picsellia_labels[label-1],
                             score)
                picsellia_rectangles.append(rectangle)

            experiment.add_evaluation(asset, rectangles=picsellia_rectangles)

    job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
    job.wait_for_done()
