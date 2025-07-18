import os
import shutil
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import ToTensorV2
from picsellia import Client
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.anchor_utils import AnchorGenerator
import sys

from exports.src.export_onnx import download_model_version

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), r'training_image\picsellia_folder'))
from training_image.picsellia_folder.dataset import PascalVOCDataset
from training_image.picsellia_folder.main import download_annotations
from tools.model_retinanet import collate_fn, build_model
from tools.retinanet_parameters import TrainingParameters, BackboneType, FPNExtraBlocks, AnchorBoxesParameters
import albumentations as A

from training_image.picsellia_folder.utils import apply_postprocess_on_predictions


def download_dataset_version(root, alias, experiment):
    annotations_folder_path = os.path.join(root, alias, 'Annotations')
    images_folder_path = os.path.join(root, alias, 'JPEGImages')

    os.makedirs(images_folder_path)
    os.makedirs(annotations_folder_path)

    dataset_version = experiment.get_dataset(alias)
    assets = dataset_version.list_assets()
    assets.download(images_folder_path, max_workers=8)

    download_annotations(dataset_version=dataset_version, annotation_folder_path=annotations_folder_path)



if __name__ == '__main__':
    dataset_root_folder = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\dataset'
    # model_weights_path = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\models\latest.pth'

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)
    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])


    model_target_path: str = 'tmp/'
    shutil.rmtree(model_target_path, ignore_errors=True)
    # experiment.get_artifact(name='model-latest').download(target_path=model_target_path)

    model_weights_path = download_model_version(model_artifact=experiment.get_artifact('model-latest'))
    # model_weights_path = 'tmp/latest.pth'

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Download datasets
    datasets = experiment.list_attached_dataset_versions()

    if not os.path.exists(dataset_root_folder):
        download_dataset_version(root=dataset_root_folder, alias='test', experiment=experiment)

    # base_model = experiment.get_base_model_version()

    # Download weights
    # download_experiment_file(base_model=base_model, experiment_file='weights')

    # total_configs = experiment.get_log('All parameters')
    #
    # # TODO change name of learning_rate parameter
    # if 'learning_rate' in parameters.keys():
    #     total_configs['learning_rate']['initial_lr'] = parameters['learning_rate']

    training_parameters = TrainingParameters(**experiment.get_log('All parameters').data)
    if len(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) == 1:
        training_parameters.anchor_boxes = AnchorBoxesParameters(
            sizes=tuple(experiment.get_log('All parameters').data['anchor_boxes_sizes']), aspect_ratios=tuple(
                [tuple(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) for i in range(5)]))
    elif len(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) == 5:
        training_parameters.anchor_boxes = AnchorBoxesParameters(
            sizes=tuple(experiment.get_log('All parameters').data['anchor_boxes_sizes']), aspect_ratios=tuple(
                experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']))

    training_parameters.device = device.type
    if device.type == 'cuda':
        training_parameters.device_name = torch.cuda.get_device_name()

    # inferences
    model = build_model(num_classes=len(experiment.get_log('LabelMap').data),
                        backbone_type=BackboneType[experiment.get_log('All parameters').data['backbone_backbone_type']],
                        add_P2_to_FPN=not experiment.get_log('All parameters').data['backbone_add_P2_to_FPN'],
                        extra_blocks_FPN=FPNExtraBlocks[experiment.get_log('All parameters').data['backbone_extra_blocks_FPN']],
                        backbone_layers_nb=experiment.get_log('All parameters').data['backbone_backbone_layers_nb'],
                        use_imageNet_pretrained_weights=training_parameters.coco_pretrained_weights,
                        trained_weights=model_weights_path,
                        score_threshold=0.2,
                        image_size=training_parameters.image_size,
                        # score_threshold=training_parameters.confidence_threshold,
                        iou_threshold=0.2,
                        unfrozen_layers=training_parameters.unfreeze,
                        mean_values=experiment.get_log('All parameters').data[
                          'augmentations_normalization_mean'],
                        std_values=experiment.get_log('All parameters').data[
                          'augmentations_normalization_std'],
                        anchor_boxes_params=training_parameters.anchor_boxes,
                        fg_iou_thresh=training_parameters.fg_iou_thresh,
                        bg_iou_thresh=training_parameters.bg_iou_thresh)

    model.anchor_generator = AnchorGenerator(
        sizes=experiment.get_log('All parameters').data['anchor_boxes_sizes'],
        aspect_ratios=experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios'])

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # dataset
    image_size = (1024, 1024)
    single_cls = True
    num_workers = 8
    batch_size = 2

    valid_transform = A.Compose([
        A.Resize(*image_size),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    val_dataset = PascalVOCDataset(
        data_folder=os.path.join(dataset_root_folder, 'test'),
        split='test',
        single_cls=single_cls,
        transform=valid_transform,
        do_class_mapping=False)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    resized_image_height, resized_image_width = image_size
    original_height = 3060
    original_width = 3060

    # calculate the scale factors for width and height
    width_scale = original_width / resized_image_width
    height_scale = original_height / resized_image_height

    metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True,
                                  max_detection_thresholds=[3000, 5000, 10000])

    for images, targets in val_data_loader:
        images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start_prediction = time.time()
        with torch.no_grad():
            predictions = model(images)
        print('Time taken for inference: {}'.format(time.time() - start_prediction))

        processed_predictions = apply_postprocess_on_predictions(
            predictions=predictions,
            iou_threshold=0.2,
            confidence_threshold=0.2)

        # send targets to GPU
        targets_gpu = [{k: v.to(device=device, non_blocking=True) for k, v in target.items()} for target in targets]

        metric.update(predictions, targets_gpu)

        for image, prediction in zip(images, predictions):
            image = image.cpu().numpy()
            image = image.transpose((1, 2, 0))

            for box in prediction['boxes']:
                box = np.round(box.cpu().numpy()).astype(int)
                x1, y1, x2, y2 = box

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            plt.imshow(image)
            plt.show()

    validation_metrics = metric.compute()
    print(f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
          f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
          f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
          f"'Precision': {float(validation_metrics['precision'][0][25][0][0][-1]):.3} / "
          f"'Recall': {float(validation_metrics['recall'][0][0][0][-1]):.3} ")
