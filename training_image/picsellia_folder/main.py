import logging
import os
import shutil
import typing
import zipfile
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Union

import albumentations as A
import picsellia
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from picsellia import Client, ModelVersion, DatasetVersion, Experiment
from picsellia.types.enums import AnnotationFileType
from pytorch_warmup import ExponentialWarmup, LinearWarmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import PascalVOCDataset, PascalVOCTestDataset
from evaluator import fill_picsellia_evaluation_tab
from tools.model_retinanet import collate_fn, build_retinanet_model, build_model
from normalize_parameters import compute_auto_normalization_parameters
from picsellia_logger import PicselliaLogger
from tools.retinanet_parameters import TrainingParameters
from trainer import train_model
from anchor_optimization.optimize_anchors_torch import compute_optimized_anchors
from utils import read_yaml_file, download_annotations
from augmentations import train_augmentation_v3, train_augmentation_v2, train_augmentation_v1

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def get_alias(dataset_version_name: str) -> str:
    '''
    Get alias (which could be "train", "test" or "val") from dataset version name. To be recognized, alias has to take
    part of dataset version name.
    :param dataset_version_name:
    :return:
    '''
    for name in ['train', 'val', 'test']:
        if name in dataset_version_name:
            return name

    raise ValueError('Unknown dataset')


def download_datasets(experiment: Experiment, root_folder: str = 'dataset'):
    '''
        .
    ├── train/
    │   ├── JPEGImages/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── Annotations/
    │   │   ├── img1.xml
    │   │   ├── img2.xml
    │   │   └── ...
    │   └── ...
    └── test/
        ├── JPEGImages/
        │   └── ...jpg
        └── Annotations/
            └── ...xml
    '''

    def download_dataset_version():
        annotations_folder_path = os.path.join(root, alias, 'Annotations')
        images_folder_path = os.path.join(root, alias, 'JPEGImages')

        os.makedirs(images_folder_path)
        os.makedirs(annotations_folder_path)

        assets = dataset_version.list_assets()
        assets.download(images_folder_path, max_workers=8)

        download_annotations(dataset_version=dataset_version, annotation_folder_path=annotations_folder_path)

    root = root_folder

    if len(experiment.list_attached_dataset_versions()) == 3:
        for alias in ['test', 'train', 'val']:
            dataset_version = experiment.get_dataset(alias)
            logging.info(f'{alias} alias for {dataset_version}')
            download_dataset_version()

    elif len(experiment.list_attached_dataset_versions()) == 2:
        for alias in ['train', 'val']:
            dataset_version = experiment.get_dataset(alias)
            logging.info(f'{alias} alias for {dataset_version}')
            download_dataset_version()


def download_experiment_file(base_model: ModelVersion, experiment_file: str):
    comport_file: bool = any([i.name == experiment_file for i in base_model.list_files()])
    if comport_file:
        base_model_weights = base_model.get_file(experiment_file)
        base_model_weights.download()


def get_advanced_config(base_model: ModelVersion) -> Union[None, dict]:
    download_experiment_file(base_model, 'config')
    config_filename = base_model.get_file('config').filename
    with open(config_filename) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def create_dataloaders(image_size: tuple[int, int], single_cls: bool, num_workers: int, batch_size: int, path_root: str,
                       augmentation_version: int, cutmix_prob: float, mixup_prob: float, mosaic_prob: float,
                       cutout_prob: float,
                       random_crop: bool = False, augmentation_hyperparams_file: typing.Optional[str] = None) -> \
        tuple[DataLoader, DataLoader, PascalVOCDataset, PascalVOCDataset]:
    if augmentation_version > 2:
        if augmentation_hyperparams_file is not None:
            augmentation_params = read_yaml_file(file_path=augmentation_hyperparams_file)
            augmentation_params = defaultdict(dict, augmentation_params)
            augmentation_params['cutout']['prob'] = cutout_prob
            augmentation_params['cutmix']['prob'] = cutmix_prob
            augmentation_params['mosaic']['prob'] = mosaic_prob
            augmentation_params['mixup']['prob'] = mixup_prob
        else:
            # augmentation_params = {}
            augmentation_params = defaultdict(dict)
            augmentation_params['cutout']['prob'] = cutout_prob
            augmentation_params['cutmix']['prob'] = cutmix_prob
            augmentation_params['mosaic']['prob'] = mosaic_prob
            augmentation_params['mixup']['prob'] = mixup_prob
            train_transform = globals()[f"train_augmentation_v{augmentation_version}"](
                random_crop=random_crop, image_size=image_size, **augmentation_params)

    else:
        train_transform = globals()[f"train_augmentation_v{augmentation_version}"](
            random_crop=random_crop, image_size=image_size)

    valid_transform = A.Compose([
        A.RandomCrop(*image_size) if random_crop else A.Resize(*image_size),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    train_dataset = PascalVOCDataset(
        data_folder=os.path.join(path_root, 'train'),
        split='train',
        single_cls=single_cls,
        transform=train_transform)

    val_dataset = PascalVOCDataset(
        data_folder=os.path.join(path_root, 'val'),
        split='train',
        single_cls=single_cls,
        transform=valid_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_data_loader, val_data_loader, train_dataset, val_dataset


def get_class_mapping_from_picsellia(dataset_versions: typing.List[DatasetVersion], single_cls: bool = False) -> \
        typing.Dict[int, str]:
    labels = ['bckg']

    if not single_cls:
        for ds_version in dataset_versions:
            for label in ds_version.list_labels():
                if label.name not in labels:
                    labels.append(label.name)

    else:
        labels.append('cls0')

    return dict(zip(range(len(labels)), labels))


def get_test_dataset_split(nb_splits_dataset: int) -> str:
    """
    Get name of test dataset depending on the number of splits in the dataset.
    If nb_splits_dataset is 2, the test dataset split will be 'val'.
    If nb_splits_dataset is 3, the test dataset split will be 'test'.
    :param nb_splits_dataset: number of splits in the dataset
    :return: name of the split used as test in the dataset
    """
    if nb_splits_dataset == 2:
        return 'val'
    else:
        return 'test'


def get_optimizer(optimizer_name: str, lr0: float, weight_decay: float, model_params) -> torch.optim.Optimizer:
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr=lr0, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model_params, lr=lr0, weight_decay=weight_decay)

    return optimizer


def flatten_dictionary(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dictionary(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


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

    optimizer = get_optimizer(optimizer_name=training_parameters.optimizer,
                              lr0=float(training_parameters.learning_rate.initial_lr),
                              weight_decay=training_parameters.weight_decay,
                              model_params=model.parameters())

    if training_parameters.clr:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            mode='triangular2',
            base_lr=float(training_parameters.learning_rate.initial_lr)/4,
            max_lr=float(training_parameters.learning_rate.initial_lr),
            step_size_up=10)

    else:
        if training_parameters.learning_rate.policy != 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=training_parameters.learning_rate.decay.step_size,
                                                           gamma=training_parameters.learning_rate.decay.gamma)
        else:
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min',
                                             factor=training_parameters.learning_rate.plateau.factor,
                                             patience=training_parameters.learning_rate.plateau.patience)

    warmup_scheduler = LinearWarmup(optimizer,
                                    warmup_period=10)

    # Logger
    picsellia_logger = PicselliaLogger.from_picsellia_client_and_experiment(picsellia_experiment=experiment,
                                                                            picsellia_client=client)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_dataset), "val": len(val_dataset)},
        title="Nb elem / split")

    picsellia_logger.log_split_table(annotations_in_split=train_dataset.number_obj_by_cls, title="Train split")
    picsellia_logger.log_split_table(annotations_in_split=val_dataset.number_obj_by_cls, title="Val split")
    picsellia_logger.on_train_begin(params=flatten_dictionary(dictionary=training_parameters.dict()),
                                    class_mapping=class_mapping)

    train_model(model=model,
                optimizer=optimizer,
                train_data_loader=train_dataloader,
                val_data_loader=val_dataloader,
                lr_scheduler=lr_scheduler,
                lr_warmup=warmup_scheduler,
                nb_epochs=training_parameters.epoch,
                path_saved_models=path_saved_models,
                loss_coefficients=training_parameters.loss.dict(),
                patience=training_parameters.patience,
                device=device,
                callback=picsellia_logger,
                mixed_precision=training_parameters.mixed_precision)

    # See predictions
    valid_transform = A.Compose([
        A.Resize(*training_parameters.image_size),
        ToTensorV2()
    ])

    test_dataset_split = get_test_dataset_split(nb_splits_dataset=len(experiment.list_attached_dataset_versions()))
    test_dataset = PascalVOCDataset(
        data_folder=os.path.join(dataset_root_folder, test_dataset_split),
        split='test',
        single_cls=True,
        transform=valid_transform)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=training_parameters.workers_number,
        batch_size=training_parameters.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    fill_picsellia_evaluation_tab(model=model,
                                  data_loader=test_dataloader,
                                  experiment=experiment,
                                  dataset_version_name=test_dataset_split,
                                  device=device)
