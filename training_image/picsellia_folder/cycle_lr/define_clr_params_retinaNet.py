import logging
import os
import sys
from enum import Enum

import numpy as np
import torch
from matplotlib import pyplot as plt
from picsellia import Client
from torch.optim import SGD
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

sys.path.append("../")
from anchor_optimization.optimize_anchors_torch import compute_optimized_anchors
from normalize_parameters import compute_auto_normalization_parameters
from tools.model_retinanet import build_retinanet_model, build_model
from tools.retinanet_parameters import TrainingParameters
from trainer import evaluate_one_epoch
from training_image.picsellia_folder.main import download_datasets, download_experiment_file, get_advanced_config, \
    create_dataloaders, get_class_mapping_from_picsellia
from utils import Averager, apply_loss_weights

try:
   from torch import GradScaler           # torch >= 2.3
except ImportError:
   from torch.cuda.amp import GradScaler  # torch < 2.3

try:
   from torch import autocast
except ImportError:
   from torch.cuda.amp import autocast

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

class POLICY(Enum):
    LINEAR = 'linear'
    LOG = 'log'


def lr_range_test(train_data_loader, val_data_loader, loss_coefficients, min_lr: 1e-7, max_lr: 1e-1, nb_epochs: int = 10,
                  policy: POLICY = POLICY.LINEAR):
    data_plot = {'lr': [],
                 'valid_loss': [],
                 'train_loss': [],
                 'accuracy': []}

    if policy == POLICY.LINEAR:
        gamma = np.linspace(min_lr, max_lr, num=nb_epochs)
    else:
        gamma = np.geomspace(min_lr, max_lr, num=nb_epochs)

    optimizer = SGD(model.parameters(), lr=min_lr)


    visualisation_val_loss = True
    best_map = 0.0
    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[3000, 5000, 10000],
                                  extended_summary=True)

    loss_training_hist = Averager()
    loss_validation_hist = Averager()

    if mixed_precision:
        scaler = GradScaler()

    for epoch in range(nb_epochs):
        loss_training_hist.reset()
        loss_validation_hist.reset()

        model.train()
        with tqdm(train_data_loader, unit="batch") as t_epoch:
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = gamma[epoch]

            for images, targets in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                images = list(image.type(torch.FloatTensor).to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                '''
                Losses: 
                - Sigmoid focal loss for classification
                - l1 for regression
                '''
                if not mixed_precision:
                    loss_dict = model(images, targets)
                    loss_dict = apply_loss_weights(loss_dict=loss_dict, loss_coefficients=loss_coefficients)

                    total_loss = sum(loss for loss in loss_dict.values())
                    total_loss_value = total_loss.item()

                    loss_training_hist.send({
                        "regression": round(loss_dict['bbox_regression'].item(), 4),
                        "classification": round(loss_dict['classification'].item(), 4),
                        "total": round(total_loss_value, 4)
                    })  # Average out the loss

                    total_loss.backward()
                    optimizer.step()

                else:
                    # Use autocast to enable mixed-precision
                    with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        loss_dict = model(images, targets)
                        loss_dict = apply_loss_weights(loss_dict=loss_dict, loss_coefficients=loss_coefficients)

                        total_loss = sum(loss for loss in loss_dict.values())
                        total_loss_value = total_loss.item()

                        loss_training_hist.send({
                            "regression": round(loss_dict['bbox_regression'].item(), 4),
                            "classification": round(loss_dict['classification'].item(), 4),
                            "total": round(total_loss_value, 4)
                        })  # Average out the loss

                    # Backward pass with scaled gradients
                    scaler.scale(total_loss).backward()

                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)

                    # Update the scaler
                    scaler.update()

                t_epoch.set_postfix(
                    total_loss=loss_training_hist.value['total'],
                    bbox_loss=loss_training_hist.value['regression'],
                    cls_loss=loss_training_hist.value['classification']
                )

            if visualisation_val_loss:
                for images, targets in val_data_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.no_grad():
                        val_loss_dict = model(images, targets)

                    # apply coefficients for regression and classification losses
                    val_loss_dict = apply_loss_weights(loss_dict=val_loss_dict, loss_coefficients=loss_coefficients)
                    total_val_loss = sum(loss for loss in val_loss_dict.values())

                    loss_validation_hist.send({
                        "regression": val_loss_dict['bbox_regression'].item(),
                        "classification": val_loss_dict['classification'].item(),
                        "total": total_val_loss.item()
                    })

        # Evaluation
        validation_metrics = evaluate_one_epoch(model, val_data_loader, device, metric)

        # TODO display precision / recall in Picsellia interface
        '''
        - ``precision``: a tensor of shape ``(TxRxKxAxM)`` containing the precision values. Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.
        - ``recall``: a tensor of shape ``(TxKxAxM)`` containing the recall values. Here ``T`` is the number of
          IoU thresholds, ``K`` is the number of classes, ``A`` is the number of areas and ``M`` is the number
          of max detections per image
        '''

        logging.info(f"Epoch #{epoch + 1} Training loss: {loss_training_hist.value} "
                     f"Validation loss {loss_validation_hist.value}"
                     f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
                     f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
                     f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
                     f"'Precision': {float(validation_metrics['precision'][0][25][0][0][-1]):.3} / "
                     f"'Recall': {float(validation_metrics['recall'][0][0][0][-1]):.3} ")
        if validation_metrics['map'] >= best_map:
            best_map = float(validation_metrics['map'])
            torch.save(model.state_dict(), os.path.join(path_saved_models, 'best.pth'))


        curr_lr = optimizer.param_groups[0]['lr']

        data_plot['train_loss'].append(loss_training_hist.value)
        data_plot['valid_loss'].append(loss_validation_hist.value)
        data_plot['lr'].append(curr_lr)
        data_plot['accuracy'].append(float(validation_metrics['map_50']))

    plt.plot(data_plot['lr'], data_plot['train_loss'], label='train_loss')
    plt.plot(data_plot['lr'], data_plot['valid_loss'], label='valid_loss')
    plt.legend()
    plt.show()

    if policy == POLICY.LOG:
        plt.semilogx(data_plot['lr'], data_plot['accuracy'], label='accuracy')
        plt.legend()
        plt.show()

    else:
        plt.semilogx(data_plot['lr'], data_plot['accuracy'], label='accuracy')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    torch.manual_seed(42)

    # Define input/output folders
    dataset_root_folder: str = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
    path_saved_models: str = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
    os.makedirs(path_saved_models, exist_ok=True)

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)
    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device.type.upper()}')

    # Download datasets
    datasets = experiment.list_attached_dataset_versions()

    # if not os.path.exists(dataset_root_folder):
    #     download_datasets(experiment=experiment, root_folder=dataset_root_folder)
    # else:
    #     logging.warning(f'A dataset was previously imported before the training.')
    download_datasets(experiment=experiment, root_folder=dataset_root_folder)

    base_model = experiment.get_base_model_version()

    # Download weights
    download_experiment_file(base_model=base_model, experiment_file='weights')

    # Download augmentation hyperparameter file
    download_experiment_file(base_model=base_model, experiment_file='aug_hyp')
    experiment.store(name='aug_hyp', path=base_model.get_file('aug_hyp').filename)

    # Download parameters
    parameters = experiment.get_log(name='parameters').data

    # Download advanced configurations from *config* file
    advanced_config = get_advanced_config(base_model=base_model)

    total_configs = parameters | advanced_config

    # TODO change name of learning_rate parameter
    if 'learning_rate' in parameters.keys():
        lr0 = parameters['learning_rate']
        del total_configs['learning_rate']

    for loss_param in ['alpha_loss', 'gamma_loss']:
        if loss_param in total_configs.keys():
            total_configs['loss'][loss_param] = total_configs[loss_param]
            del total_configs[loss_param]

    training_parameters = TrainingParameters(**total_configs)

    if 'learning_rate' in parameters.keys():
        training_parameters.learning_rate.initial_lr = parameters['learning_rate']

    training_parameters.device = device.type
    if device.type == 'cuda':
        training_parameters.device_name = torch.cuda.get_device_name()

    if training_parameters.augmentations.normalization.auto_norm:
        normalization_parameters = compute_auto_normalization_parameters(image_folder=dataset_root_folder)
        # overwrite normalization parameters
        training_parameters.augmentations.normalization.mean = normalization_parameters['RGB mean']
        training_parameters.augmentations.normalization.std = normalization_parameters['RGB std']

    # Create dataloaders
    train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(
        image_size=training_parameters.image_size,
        single_cls=training_parameters.single_class,
        num_workers=training_parameters.workers_number,
        batch_size=training_parameters.batch_size,
        path_root=dataset_root_folder,
        random_crop=training_parameters.augmentations.crop,
        augmentation_version=training_parameters.augmentations.version,
        cutmix_prob=float(parameters['cutmix']) if 'cutmix' in parameters.keys() else 0.0,
        mosaic_prob=float(parameters['mosaic']) if 'mosaic' in parameters.keys() else 0.0,
        cutout_prob=float(parameters['cutout']) if 'cutout' in parameters.keys() else 0.0,
        mixup_prob=float(parameters['mixup']) if 'mixup' in parameters.keys() else 0.0,

    )

    class_mapping = get_class_mapping_from_picsellia(dataset_versions=datasets,
                                                     single_cls=training_parameters.single_class)

    if training_parameters.anchor_boxes.auto_size:
        """
        # first version of the anchor boxes optimizer
        anchor_boxes_optimizer = AnchorBoxOptimizer(dataloader=train_dataloader,
                                                    add_P2_to_FPN=training_parameters.backbone.add_P2_to_FPN)
        anchor_sizes = anchor_boxes_optimizer.get_anchor_boxes_sizes()
        training_parameters.anchor_boxes.sizes = anchor_sizes
        # (0.5, 1.0, 2.0) is RetinaNet aspect_ratio by default
        training_parameters.anchor_boxes.aspect_ratios = tuple([(0.5, 1, 2) for i in range(len(anchor_sizes))])
        """

        anchors_parameters = compute_optimized_anchors(
            annotations_path=os.path.join(dataset_root_folder, 'train', 'Annotations'),
            image_size=training_parameters.image_size,
            temp_csv_filepath='labels.csv')

        training_parameters.anchor_boxes.sizes = anchors_parameters['sizes']
        training_parameters.anchor_boxes.aspect_ratios = [float(r) for r in anchors_parameters['ratios']]

    # Build model
    if training_parameters.backbone.backbone_layers_nb == 50 and training_parameters.version == 2:
        model = build_retinanet_model(num_classes=len(class_mapping),
                                      use_COCO_pretrained_weights=training_parameters.coco_pretrained_weights,
                                      score_threshold=training_parameters.confidence_threshold,
                                      iou_threshold=training_parameters.iou_threshold,
                                      unfrozen_layers=training_parameters.unfreeze,
                                      mean_values=training_parameters.augmentations.normalization.mean,
                                      std_values=training_parameters.augmentations.normalization.std,
                                      anchor_boxes_params=training_parameters.anchor_boxes,
                                      fg_iou_thresh=training_parameters.fg_iou_thresh,
                                      bg_iou_thresh=training_parameters.bg_iou_thresh,
                                      image_size=training_parameters.image_size,
                                      alpha_loss=training_parameters.loss.alpha_loss,
                                      gamma_loss=training_parameters.loss.gamma_loss)

    else:
        model = build_model(num_classes=len(class_mapping),
                            use_imageNet_pretrained_weights=training_parameters.coco_pretrained_weights,
                            score_threshold=training_parameters.confidence_threshold,
                            iou_threshold=training_parameters.iou_threshold,
                            unfrozen_layers=training_parameters.unfreeze,
                            mean_values=training_parameters.augmentations.normalization.mean,
                            std_values=training_parameters.augmentations.normalization.std,
                            anchor_boxes_params=training_parameters.anchor_boxes,
                            fg_iou_thresh=training_parameters.fg_iou_thresh,
                            bg_iou_thresh=training_parameters.bg_iou_thresh,
                            backbone_type=training_parameters.backbone.backbone_type,
                            backbone_layers_nb=training_parameters.backbone.backbone_layers_nb,
                            add_P2_to_FPN=training_parameters.backbone.add_P2_to_FPN,
                            extra_blocks_FPN=training_parameters.backbone.extra_blocks_FPN,
                            image_size=training_parameters.image_size,
                            alpha_loss=training_parameters.loss.alpha_loss,
                            gamma_loss=training_parameters.loss.gamma_loss
                            )

    model.to(device)

    mixed_precision = True

    min_lr = 1e-7
    max_lr = 0.001
    nb_epochs = 10

    lr_range_test(train_data_loader=train_dataloader,
                  val_data_loader=val_dataloader,
                  loss_coefficients=training_parameters.loss.dict(),
                  min_lr=min_lr,
                  max_lr=max_lr)