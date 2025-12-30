import os
import shutil
import zipfile
from typing import List, Dict

import numpy as np
import picsellia
import torch
from matplotlib import pyplot as plt
from picsellia import Experiment, Client
from picsellia.types.enums import InferenceType, LogType, AnnotationFileType
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

from dataset import PascalVOCDataset
from tools.model_retinanet import build_retinanet_model, collate_fn
from tools.retinanet_parameters import TrainingParameters


def plot_precision_recall_curve(validation_metrics: Dict, recall_thresholds: List[float]) -> plt.plot:
    """
    Plot precision-recall curve depending on validation metrics.
    :param validation_metrics: validation COCO metrics at the end of the training
    :param recall_thresholds: recall thresholds used to compute the validation metric_torch_model
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


def fill_picsellia_evaluation_tab(model: RetinaNet, data_loader: torch.utils.data.DataLoader, experiment: Experiment,
                                  dataset_version_name: str, device) -> None:
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

    def log_final_metrics(results: Dict[str, float]) -> None:
        data = {
            "mAP": f'{float(results["map"]):.3}',
            'mAP[50]': f'{float(results["map_50"]): .3}',
            'mAP[75]': f'{float(results["map_75"]):.3}',
            'Precision': f'{float(results["precision"][0][25][0][0][-1]):.3}',
            'Recall': f'{float(results["recall"][0][0][0][-1]):.3}'
        }
        print(data)
        experiment.log(name='Final results', type=LogType.TABLE, data=data)

    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[3000, 5000, 10000],
                                  extended_summary=True)

    dataset_version = experiment.get_dataset(name=dataset_version_name)
    picsellia_labels = dataset_version.list_labels()

    model.topk_candidates = 5000
    model.detections_per_img = 3000

    model.to(device)
    model.eval()

    for i, (images, targets, file_paths) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            metric.update(predictions, targets)
            # print(metric.compute()['map_50'])

        for idx in range(len(images)):
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
                             picsellia_labels[label - 1],
                             score)
                picsellia_rectangles.append(rectangle)

            experiment.add_evaluation(asset, rectangles=picsellia_rectangles)
    log_final_metrics(results=metric.compute())

    job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
    # job.wait_for_done()


if __name__ == '__main__':
    import albumentations as A
    from albumentations import ToTensorV2

    client = Client(api_token=os.environ['api_token'], organization_name='SGS_France')
    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # download_dataset_version(root='./dataset', alias='val', experiment=experiment)

    model_weights_path = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\models\mix_ft.pth'
    test_dataset_path = r'./dataset/val'
    training_parameters = TrainingParameters(**experiment.get_log('All parameters').data)
    training_parameters.device = device.type
    if device.type == 'cuda':
        training_parameters.device_name = torch.cuda.get_device_name()

    # inferences
    model = build_retinanet_model(num_classes=2,
                                  use_COCO_pretrained_weights=training_parameters.coco_pretrained_weights,
                                  trained_weights=model_weights_path,
                                  score_threshold=0.25,
                                  # score_threshold=training_parameters.confidence_threshold,
                                  iou_threshold=0.1,
                                  unfrozen_layers=training_parameters.unfreeze,
                                  mean_values=experiment.get_log('All parameters').data[
                                      'augmentations_normalization_mean'],
                                  std_values=experiment.get_log('All parameters').data[
                                      'augmentations_normalization_std'],
                                  anchor_boxes_params=training_parameters.anchor_boxes,
                                  fg_iou_thresh=training_parameters.fg_iou_thresh,
                                  bg_iou_thresh=training_parameters.bg_iou_thresh,
                                  image_size=(2048, 2048)
                                  )
    model.anchor_generator = AnchorGenerator(
        sizes=experiment.get_log('All parameters').data['anchor_boxes_sizes'],
        aspect_ratios=experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios'])

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # dataset
    # image_size = experiment.get_log('All parameters').data['image_size']
    image_size = (2048, 2048)
    random_crop = False

    valid_transform = A.Compose([
        A.RandomCrop(*image_size) if random_crop else A.Resize(*image_size),
        ToTensorV2()
        ], bbox_params = A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    # test_dataset = PascalVOCTestDataset(
    #     image_folder=r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\dataset\test\JPEGImages',
    #     transform=valid_transform
    # )

    test_dataset = PascalVOCDataset(
        data_folder= test_dataset_path,
        split='test',
        single_cls=True,
        transform=valid_transform)

    val_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=8,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    fill_picsellia_evaluation_tab(model=model, data_loader=val_data_loader, experiment=experiment,
                                  dataset_version_name='val', device=device)
