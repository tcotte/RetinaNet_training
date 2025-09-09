import logging
import math
import os
from distutils.util import strtobool
from typing import List, Dict, Optional

import albumentations as A
import numpy as np
import onnxruntime
import picsellia
import requests
import torch
import tqdm
from PIL import Image
from albumentations import ToTensorV2
from picsellia import Asset, DatasetVersion, ModelVersion, Annotation, Label, Experiment
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.sdk.asset import MultiAsset
from picsellia.types.enums import InferenceType
from torchvision.models.detection import RetinaNet

from tools.model_retinanet import build_model, build_retinanet_model
from tools.picsellia_utils import download_model_version
from tools.retinanet_parameters import BackboneType, FPNExtraBlocks, AnchorBoxesParameters


class PreAnnotator:
    def __init__(self, client, dataset_version_id, model_version_id, parameters: dict, img_size: int, model_type: str):
        self.model: Optional[RetinaNet] = None
        self._is_onnx = True if model_type == 'onnx' else False
        self.client = client
        self.dataset_version: DatasetVersion = client.get_dataset_version_by_id(dataset_version_id)
        self.model_version: ModelVersion = client.get_model_version_by_id(model_version_id)
        self.parameters = parameters

        self.model_labels = []
        self.dataset_labels = []
        self.model_info = {}
        self.model_name = "model-latest" if not self._is_onnx else "model_latest_onnx"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_size = (img_size, img_size)
        self.max_det = parameters.get("max_det", 500)

        # self.dataset_version.delete_all_annotations()

        self._single_class = parameters.get("single_class", False)

        if not 'replace' in parameters.keys():
            self._replace_annotation = True
        else:
            self._replace_annotation: bool = bool(strtobool(parameters["replace"]))

        logging.info(f'Replace annotation argument has the value {self._replace_annotation}')

        self._labelmap = self._get_labelmap(dataset_version=self.dataset_version)

    def setup_pre_annotation_job(self):
        """
        Set up the pre-annotation job by performing various checks and preparing the model.
        """
        logging.info("Setting up pre-annotation job...")
        self._model_sanity_check()

        if self.dataset_version.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_type_to_model_type()
            self._create_labels_in_dataset()

        else:
            self._create_labels_in_dataset()

        # else:
        #     self._type_coherence_check()
        #     self._labels_coherence_check()

        # self._download_model_weights()
        # self._load_yolov8_model()
        self._load_retinaNet_model()

    def _get_not_annotated_assets(self) -> MultiAsset:
        """
        Get a multi-asset which lists all assets that have not already been annotated
        :return: list of all assets that have not already been annotated
        """
        multi_assets = self.dataset_version.list_assets()
        list_annotated_assets: list[bool] = [asset.list_annotations() != [] for asset in multi_assets]

        # debug logs
        # for asset in multi_assets:
        #     logging.info(f"Asset annotation: {asset.list_annotations()}")
        # logging.info(f'not_ann_multi_asset_variable: {list_annotated_assets}')

        if sum(list_annotated_assets) < len(list_annotated_assets):
            for index in list_annotated_assets:
                if index:
                    multi_assets.pop(i=index)

        return multi_assets

    def get_validation_transform(self):
        return A.Compose([
            A.Resize(*self.inference_size),
            ToTensorV2()
        ])

    def pre_annotate(self, confidence_threshold: float = 0.25):
        """
        Processes and annotates assets in the dataset using the YOLOv8 model.

        Args:
            confidence_threshold (float, optional): A threshold value used to filter
                                                    the bounding boxes based on their
                                                    confidence scores. Only boxes with
                                                    confidence scores above this threshold
                                                    are annotated. Defaults to 0.5.
        """
        if self._replace_annotation:
            dataset_size = self.dataset_version.sync()["size"]
            multi_assets = self.dataset_version.list_assets()

        else:
            multi_assets = self._get_not_annotated_assets()
            dataset_size = len(multi_assets)
            logging.info(f"Found {dataset_size} not annotated assets over {self.dataset_version.sync()['size']} "
                         f"assets.")

        batch_size = self.parameters.get("batch_size", 4)
        iou_threshold = self.parameters.get("iou", 0.7)
        confidence_threshold = self.parameters.get("confidence", 0.25)
        batch_size = min(batch_size, dataset_size)

        total_batch_number = math.ceil(dataset_size / batch_size)

        logging.info(
            f"\n-- Starting processing {total_batch_number} batch(es) of {batch_size} image(s) | "
            f"Total images: {dataset_size} --"
        )

        validation_transform = self.get_validation_transform()

        for batch_number in tqdm.tqdm(
                range(total_batch_number),
                desc="Processing batches",
                unit="batch",
        ):
            assets = multi_assets[batch_number * batch_size:(batch_number + 1) * batch_size]
            url_list = [asset.sync()["data"]["presigned_url"] for asset in assets]

            images = [Image.open(requests.get(url, stream=True).raw) for url in url_list]
            transformed_images = [validation_transform(image=np.array(image))['image'] / 255. for image in images]
            transformed_images = [image.to(self.device) for image in transformed_images]

            if not self._is_onnx:
                with torch.no_grad():
                    predictions = self.model(transformed_images)
            else:
                # do multiple inference because the current exported onnx graph does not have dynamic axes (see
                # https://github.com/microsoft/onnxruntime/issues/9867)
                predictions = []
                for image in transformed_images:
                    ort_inputs = {
                        self._session.get_inputs()[0].name: np.expand_dims(self.to_numpy(image), 0)}

                    prediction = self._session.run(None, ort_inputs)
                    predictions.append({'boxes': prediction[0], 'scores': prediction[1], 'labels': prediction[2]})

                # predictions = [self.model(Image.open(requests.get(url, stream=True).raw), iou=iou_threshold,
                #                           imgsz=self.inference_size,
                #                           max_det=self.max_det,
                #                           conf=confidence_threshold) for url in url_list]
                # predictions = self.flatten(predictions)

            for asset, prediction in list(zip(assets, predictions)):
                # if len(asset.list_annotations()) == 0:
                if len(prediction) > 0:
                    if self.dataset_version.type == InferenceType.OBJECT_DETECTION:
                        self._format_and_save_rectangles(asset, prediction, confidence_threshold)

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    @staticmethod
    def flatten(xss):
        return [x for xs in xss for x in xs]

    def _get_label_by_name(self, labelmap: Dict[str, Label], label_name: str) -> Label:
        if label_name not in labelmap:
            raise ValueError(f"The label {label_name} does not exist in the labelmap.")

        return labelmap[label_name]

    def _get_labelmap(self, dataset_version: DatasetVersion) -> Dict[str, Label]:
        labelmap = {label.name: label for label in dataset_version.list_labels()}

        if len(labelmap) != 0:
            return labelmap

        # labelmap was not found in model version, we try to retrieve it from experiment
        else:
            experiment = self._get_experiment_attached_to_model_version()
            try:
                return experiment.get_log('LabelMap').data

            except Exception as e:
                logging.warning(f'Could not retrieve labelmap for experiment: {str(e)}')

    @staticmethod
    def _reset_annotations(asset) -> None:
        """
        Erase current annotations of an asset sent as parameter.
        :param asset: asset in which the annotations will be removed
        """
        if asset.list_annotations() != []:
            # update without any annotation
            asset.get_annotation().delete()

    def get_inference_asset_ratio(self, asset: Asset) -> tuple[float, float]:
        """
        Get the ratio between the data size from asset and the inference size to be able to reframe the bounding boxes
        to data coordinates.
        :param asset: asset from which the ratio will be calculated
        :return: ratio between the data width and the inference width, and the ratio between the data height and the
        inference height.
        """
        width_scale = asset.width / self.inference_size[0]
        height_scale = asset.height / self.inference_size[1]
        return width_scale, height_scale

    def _format_and_save_rectangles(self, asset: Asset, prediction: dict,
                                    confidence_threshold: float = 0.25) -> None:
        # remove current annotations
        self._reset_annotations(asset)

        boxes = prediction['boxes'].cpu().numpy() if isinstance(prediction['boxes'], torch.Tensor) else prediction[
            'boxes']
        scores = prediction['scores'].cpu().numpy() if isinstance(prediction['boxes'], torch.Tensor) else (
            prediction)['scores']
        labels = prediction['labels'].cpu().numpy().astype(np.int16) if isinstance(prediction['labels'], torch.Tensor) \
            else prediction['labels']

        width_scale, height_scale = self.get_inference_asset_ratio(asset)

        #  Convert predictions to Picsellia format
        rectangle_list: list = []
        nb_box_limit = self.max_det
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        if len(boxes) > 0:
            annotation: Annotation = asset.create_annotation(duration=0.0)
        else:
            return
        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                try:
                    if not self._single_class:
                        label = list(self._labelmap.values())[labels[i]]
                    else:
                        label = next(iter(self._labelmap.values()))
                    e = boxes[i].tolist()
                    box = [
                        int(e[0] * width_scale),
                        int(e[1] * height_scale),
                        int((e[2] - e[0]) * width_scale),
                        int((e[3] - e[1]) * height_scale),
                    ]
                    box.append(label)
                    rectangle_list.append(tuple(box))
                except ResourceNotFoundError as e:
                    print(e)
                    continue

        if len(rectangle_list) > 0:
            annotation.create_multiple_rectangles(rectangle_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    @staticmethod
    def round_to_multiple(number: int, multiple: int) -> int:
        """
        Round some number to it's nearest multiple.
        Ex : number 99 / multiple 25 -> return 100
        """
        return multiple * round(number / multiple)

    def _model_sanity_check(self) -> None:
        """
        Perform a sanity check on the model.
        """
        self._check_model_file_integrity()
        self._validate_model_inference_type()
        logging.info(f"Model {self.model_version.name} passed sanity checks.")

    def _check_model_file_integrity(self) -> None:
        """
        Check the integrity of the model file by verifying it exists as "model-latest" and is an ONNX model.

        Raises:
            ResourceNotFoundError: If the model file is not an ONNX file.
        """
        model_file = self.model_version.get_file(self.model_name)

        if model_file.filename.endswith('.zip'):
            model_filename = download_model_version(model_artifact=model_file, model_target_path='models')
            self._model_weights_path = model_filename

        else:
            model_file.download(target_path='models')
            model_filename = model_file.filename
            self._model_weights_path = os.path.join('models', model_filename)

        if model_filename.endswith(".pt") or model_filename.endswith(".pth"):
            self._is_onnx = False
        elif model_file.filename.endswith(".onnx"):
            self._is_onnx = True
        else:
            raise ResourceNotFoundError("Model file must be a pt/pth/onnx file.")

    def _validate_model_inference_type(self) -> None:
        """
        Validate the model's inference type.

        Raises:
            PicselliaError: If the model type is not configured.
        """
        if self.model_version.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError("Model type is not configured.")

    def _set_dataset_type_to_model_type(self) -> None:
        """
        Set the dataset type to the model type.
        """
        self.dataset_version.set_type(self.model_version.type)
        logging.info(f"Dataset type set to {self.model_version.type}")

    def _create_labels_in_dataset(self) -> None:
        """
        Creates labels in the dataset based on the model's labels. It first retrieves the model's labels,
        then creates corresponding labels in the dataset version if they do not already exist.

        This method updates the 'model_labels' and 'dataset_labels' attributes of the class with the
        labels from the model (if they don't already exist) and the labels currently in the dataset, respectively.
        """
        if not self.model_labels:
            self.model_labels = self._get_model_labels()

        for label in tqdm.tqdm(self.model_labels):
            try:
                self.dataset_version.create_label(name=label)
            except picsellia.exceptions.ResourceConflictError:
                logging.info(f'Label {label} already created')

        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]
        logging.info(f"Labels created in dataset: {self.dataset_labels}")

        # refresh labelmap
        self._labelmap = self._get_labelmap(dataset_version=self.dataset_version)

    def _validate_dataset_and_model_type(self) -> None:
        """
        Validate that the dataset type matches the model type.

        Raises:
            PicselliaError: If the dataset type does not match the model type.
        """
        if self.dataset_version.type != self.model_version.type:
            raise PicselliaError(
                f"Dataset type {self.dataset_version.type} does not match model type {self.model_version.type}."
            )

    def _validate_label_overlap(self) -> None:
        """
        Validate that there is an overlap between model labels and dataset labels.

        Raises:
            PicselliaError: If no overlapping labels are found.
        """
        self.model_labels = self._get_model_labels()
        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]

        model_labels_set = set(self.model_labels)
        dataset_labels_set = set(self.dataset_labels)

        overlapping_labels = model_labels_set.intersection(dataset_labels_set)
        non_overlapping_dataset_labels = dataset_labels_set - model_labels_set

        if not overlapping_labels:
            raise PicselliaError(
                "No overlapping labels found between model and dataset. "
                "Please check the labels between your dataset version and your model.\n"
                f"Dataset labels: {self.dataset_labels}\n"
                f"Model labels: {self.model_labels}"
            )

        # Log the overlapping labels
        logging.info(f"Using labels: {list(overlapping_labels)}")

        if non_overlapping_dataset_labels:
            logging.info(
                f"WARNING: Some dataset version's labels are not present in model "
                f"and will be skipped: {list(non_overlapping_dataset_labels)}"
            )

    # def _download_model_weights(self) -> None:
    #     """
    #     Download the model weights and save it in `self.model_weights_path`.
    #     """
    #     model_weights = self.model_version.get_file(self.model_name)
    #     model_weights.download()
    #     self.model_weights_path = os.path.join(os.getcwd(), model_weights.filename)
    #     logging.info(f"Model weights {model_weights.filename} downloaded successfully")
    #
    # def _load_yolov8_model(self) -> None:
    #     try:
    #         self.model = YOLO(self.model_weights_path)
    #         logging.info("Model loaded in memory.")
    #     except Exception as e:
    #         raise PicselliaError(f"Impossible to load saved model located at: {self.model_weights_path}")

    def _get_experiment_attached_to_model_version(self) -> Experiment:
        return self.client.get_experiment_by_id(self.model_version.get_context().experiment_id)

    def _load_retinaNet_model(self) -> None:
        if not self._is_onnx:
            try:
                experiment = self._get_experiment_attached_to_model_version()

                if (len(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) == 1 or
                        len(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) == 3):
                    anchor_boxes = AnchorBoxesParameters(
                        sizes=tuple(experiment.get_log('All parameters').data['anchor_boxes_sizes']),
                        aspect_ratios=tuple(
                            [tuple(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) for i in
                             range(5)]))
                elif len(experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']) == 5:
                    anchor_boxes = AnchorBoxesParameters(
                        sizes=tuple(experiment.get_log('All parameters').data['anchor_boxes_sizes']),
                        aspect_ratios=tuple(
                            experiment.get_log('All parameters').data['anchor_boxes_aspect_ratios']))

                if experiment.get_log('All parameters').data['version'] == 2:
                    self.model = build_retinanet_model(num_classes=len(experiment.get_log('LabelMap').data),
                                             use_COCO_pretrained_weights=False,
                                             trained_weights=self._model_weights_path,
                                             score_threshold=0.2,
                                             image_size=self.inference_size,
                                             # score_threshold=training_parameters.confidence_threshold,
                                             iou_threshold=0.2,
                                             unfrozen_layers=3,
                                             mean_values=experiment.get_log('All parameters').data[
                                                 'augmentations_normalization_mean'],
                                             std_values=experiment.get_log('All parameters').data[
                                                 'augmentations_normalization_std'],
                                             anchor_boxes_params=anchor_boxes,
                                             fg_iou_thresh=experiment.get_log('All parameters').data['fg_iou_thresh'],
                                             bg_iou_thresh=experiment.get_log('All parameters').data['bg_iou_thresh']
                                             )
                else:

                    self.model = build_model(num_classes=len(experiment.get_log('LabelMap').data),
                                             backbone_type=BackboneType[
                                                 experiment.get_log('All parameters').data['backbone_backbone_type']],
                                             add_P2_to_FPN=not experiment.get_log('All parameters').data[
                                                 'backbone_add_P2_to_FPN'],
                                             extra_blocks_FPN=FPNExtraBlocks[
                                                 experiment.get_log('All parameters').data['backbone_extra_blocks_FPN']],
                                             backbone_layers_nb=experiment.get_log('All parameters').data[
                                                 'backbone_backbone_layers_nb'],
                                             use_imageNet_pretrained_weights=False,
                                             trained_weights=self._model_weights_path,
                                             score_threshold=0.2,
                                             image_size=self.inference_size,
                                             # score_threshold=training_parameters.confidence_threshold,
                                             iou_threshold=0.2,
                                             unfrozen_layers=3,
                                             mean_values=experiment.get_log('All parameters').data[
                                                 'augmentations_normalization_mean'],
                                             std_values=experiment.get_log('All parameters').data[
                                                 'augmentations_normalization_std'],
                                             anchor_boxes_params=anchor_boxes,
                                             fg_iou_thresh=experiment.get_log('All parameters').data['fg_iou_thresh'],
                                             bg_iou_thresh=experiment.get_log('All parameters').data['bg_iou_thresh']
                                             )

                self.model.eval()
                self.model.to(self.device)

            except Exception as e:
                raise PicselliaError(f"Impossible to load saved model located at: {self._model_weights_path}. "
                                     f"Error message: {e} ")

        else:
            self._session = onnxruntime.InferenceSession(self._model_weights_path, providers=["CUDAExecutionProvider"])

    def _get_model_labels(self) -> List[str]:
        """
        Get the labels from the model.

        Returns:
            list[str]: A list of label names from the model.
        Raises:
            InsufficientResourcesError: If no labels are found or if labels are not in dictionary format.
        """
        self.model_info = self.model_version.sync()
        if "labels" not in self.model_info:
            raise InsufficientResourcesError(
                f"No labels found for model {self.model_version.name}."
            )

        if not isinstance(self.model_info["labels"], dict):
            raise InsufficientResourcesError(
                "Model labels must be in dictionary format."
            )

        return list(self.model_info["labels"].values())
