import numpy as np
import torch
from matplotlib import pyplot as plt
from picsellia import Experiment
from picsellia.types.enums import InferenceType


def plot_precision_recall_curve(validation_metrics: dict, recall_thresholds: list[float]) -> plt.plot:
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


def fill_picsellia_evaluation_tab(model: torch.nn.Module, data_loader, experiment: Experiment,
                                  dataset_version_name: str,
                                  device, batch_size: int):
    dataset_version = experiment.get_dataset(name=dataset_version_name)
    picsellia_labels = dataset_version.list_labels()

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
                             picsellia_labels[label],
                             score)
                picsellia_rectangles.append(rectangle)

            evaluation = experiment.add_evaluation(asset, rectangles=picsellia_rectangles)

            job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
    job.wait_for_done()
