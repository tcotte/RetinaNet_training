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

from tqdm import tqdm

from exports.src.export_onnx import download_model_version

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), r'training_image\picsellia_folder'))
from training_image.picsellia_folder.dataset import PascalVOCDataset, PascalVOCTestDataset
from training_image.picsellia_folder.main import download_annotations
from tools.model_retinanet import collate_fn, build_model, build_retinanet_model
from tools.retinanet_parameters import TrainingParameters, BackboneType, FPNExtraBlocks, AnchorBoxesParameters
import albumentations as A

from training_image.picsellia_folder.utils import apply_postprocess_on_predictions



if __name__ == '__main__':
    # dataset_root_folder = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\CA\dataset'
    # model_weights_path = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\models\latest.pth'

    deployment_id = '0199e83c-c727-7bc6-85de-3a1f413183b9'
    dataset_folder = 'dataset/FT'

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)

    champion_experiment_id = '0199f267-2f41-7643-bdae-a83280d92713'
    shadow_experiment_id = '019a29f4-460d-7486-a795-75e615e55746'

    # Get experiment
    for model_type, experiment_id in zip(['champion', 'shadow'], [champion_experiment_id, shadow_experiment_id]):
        experiment = client.get_experiment_by_id(id=experiment_id)

        model_target_path: str = f'tmp/{model_type}/'
        shutil.rmtree(model_target_path, ignore_errors=True)
        # experiment.get_artifact(name='model-latest').download(target_path=model_target_path)

        # model_weights_path = download_model_version(model_artifact=experiment.get_artifact('model-latest'),
        #                                             model_target_path=f'tmp/models/{model_type}')
        model_weights_path: str = f'tmp/models/{model_type}/latest.pth'

        # model_weights_path = 'tmp/latest.pth'

        # Get device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Download datasets
        datasets = experiment.list_attached_dataset_versions()

        if not os.path.exists(dataset_folder):
            deployment = client.get_deployment_by_id(deployment_id)
            assets = deployment.list_predicted_assets()
            for asset in assets:
                asset.download(dataset_folder)

        # base_model = experiment.get_base_model_version()

        # Download weights
        # download_experiment_file(base_model=base_model, experiment_file='weights')

        # total_configs = experiment.get_log('All parameters')
        #
        # # TODO change name of learning_rate parameter
        # if 'learning_rate' in parameters.keys():
        #     total_configs['learning_rate']['initial_lr'] = parameters['learning_rate']

        training_parameters = TrainingParameters(**experiment.get_log('All parameters').data)

        if 'anchor_boxes_sizes' in experiment.get_log('All parameters').data:
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
        globals()[f'model_{model_type}'] = build_retinanet_model(num_classes=len(experiment.get_log('labelmap').data),
                            trained_weights=model_weights_path,
                            score_threshold=0.3,
                            image_size=training_parameters.image_size,
                            # score_threshold=training_parameters.confidence_threshold,
                            iou_threshold=0.2,
                            unfrozen_layers=training_parameters.unfreeze,
                            mean_values=experiment.get_log('All parameters').data[
                              'augmentations_normalization_mean'],
                            std_values=experiment.get_log('All parameters').data[
                              'augmentations_normalization_std'],
                            fg_iou_thresh=training_parameters.fg_iou_thresh,
                            bg_iou_thresh=training_parameters.bg_iou_thresh,
                            max_det=5000)

        if 'anchor_boxes_sizes' in experiment.get_log('All parameters').data:
            globals()[f'model_{model_type}'].anchor_generator = AnchorGenerator(
                sizes=experiment.get_log('All parameters').data['anchor_boxes_sizes'],
                aspect_ratios=experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios'])


        globals()[f'model_{model_type}'].load_state_dict(torch.load(model_weights_path))
        globals()[f'model_{model_type}'].to(device)
        globals()[f'model_{model_type}'].eval()

    # dataset
    image_size = (2048, 2048)
    single_cls = True
    num_workers = 8
    batch_size = 1

    valid_transform = A.Compose([
        A.Resize(*image_size),
        ToTensorV2()
    ])

    val_dataset = PascalVOCTestDataset(
        image_folder=dataset_folder,
        transform=valid_transform)

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

    # metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True,
    #                               max_detection_thresholds=[3000, 5000, 10000])
    #
    # for images, targets in val_data_loader:
    #     images = list(image.to(device) for image in images)
    #     # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    #     start_prediction = time.time()
    #     with torch.no_grad():
    #         predictions = model(images)
    #     print('Time taken for inference: {}'.format(time.time() - start_prediction))
    #
    #     processed_predictions = apply_postprocess_on_predictions(
    #         predictions=predictions,
    #         iou_threshold=0.2,
    #         confidence_threshold=0.2)
    #
    #     # send targets to GPU
    #     targets_gpu = [{k: v.to(device=device, non_blocking=True) for k, v in target.items()} for target in targets]
    #
    #     metric.update(predictions, targets_gpu)
    #
    #     for image, prediction in zip(images, predictions):
    #         image = image.cpu().numpy()
    #         image = image.transpose((1, 2, 0))
    #
    #         for box in prediction['boxes']:
    #             box = np.round(box.cpu().numpy()).astype(int)
    #             x1, y1, x2, y2 = box
    #
    #             cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #
    #         plt.imshow(image)
    #         plt.show()
    #
    # validation_metrics = metric.compute()
    # print(f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
    #       f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
    #       f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
    #       f"'Precision': {float(validation_metrics['precision'][0][25][0][0][-1]):.3} / "
    #       f"'Recall': {float(validation_metrics['recall'][0][0][0][-1]):.3} ")

    metric_champion_model = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True,
                                              max_detection_thresholds=[3000, 5000, 10000])
    metric_shadow_model = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True,
                                             max_detection_thresholds=[3000, 5000, 10000])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # compute ONNX Runtime output prediction
    nb_img = 0
    for images, image_name in tqdm(val_data_loader):
        output_image = cv2.imread(os.path.join(dataset_folder, image_name[0]))[:, :, ::-1]

        images = list(image.to(device) for image in images)


        with torch.no_grad():
            shadow_outputs = model_shadow(images)

        with torch.no_grad():
            champion_outputs = model_champion(images)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        output_champion = cv2.resize(output_image.copy(), image_size)
        # output_champion = np.transpose(output_champion, (1, 2, 0))
        boxes_champion = champion_outputs[0]['boxes'].cpu().numpy().astype(int)
        for i in boxes_champion:
            cv2.rectangle(output_champion, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)

        ax[0].imshow(output_champion)

        output_shadow = cv2.resize(output_image.copy(), image_size)
        # output_shadow = np.transpose(output_shadow, (1, 2, 0))
        boxes_shadow = shadow_outputs[0]['boxes'].cpu().numpy().astype(int)
        for i in boxes_shadow:
            cv2.rectangle(output_shadow, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)

        ax[1].imshow(output_shadow)

        ax[0].set_title('Champion inference \n'
                        f'Number of predictions: {len(boxes_champion)}')

        ax[1].set_title('Shadow inference \n'
                        f'Number of predictions: {len(boxes_shadow)}')

        for i in ax:
            i.set_axis_off()

        nb_img += 1

        plt.savefig(rf'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\output\inference_{nb_img}.png')

    #     metric_champion_model.update([champion_outputs], targets_gpu)
    #     metric_shadow_model.update([shadow_outputs], targets_gpu)
    #
    #
    #
    # champion_model_validation_metrics = metric_champion_model.compute()
    # print("Torch model: \n"
    #       f"- Accuracies: 'mAP' {float(champion_model_validation_metrics['map']):.3} / "
    #       f"'mAP[50]': {float(champion_model_validation_metrics['map_50']):.3} / "
    #       f"'mAP[75]': {float(champion_model_validation_metrics['map_75']):.3} /"
    #       f"'Precision': {float(champion_model_validation_metrics['precision'][0][25][0][0][-1]):.3} / "
    #       f"'Recall': {float(champion_model_validation_metrics['recall'][0][0][0][-1]):.3} ")
    #
    # shadow_model_validation_metrics = metric_shadow_model.compute()
    # print("ONNX model: \n"
    #       f"- Accuracies: 'mAP' {float(shadow_model_validation_metrics['map']):.3} / "
    #       f"'mAP[50]': {float(shadow_model_validation_metrics['map_50']):.3} / "
    #       f"'mAP[75]': {float(shadow_model_validation_metrics['map_75']):.3} /"
    #       f"'Precision': {float(shadow_model_validation_metrics['precision'][0][25][0][0][-1]):.3} / "
    #       f"'Recall': {float(shadow_model_validation_metrics['recall'][0][0][0][-1]):.3} ")
