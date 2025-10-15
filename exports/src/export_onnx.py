import logging
import os
import sys
from typing import Optional

import picsellia
import torch
from picsellia import Client
from picsellia.types.enums import JobRunStatus

from tools.picsellia_utils import download_model_version

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), r'training_image\picsellia_folder'))
from tools.model_retinanet import build_retinanet_model

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def get_model_version_labelmap(experiment) -> Optional[dict]:
    for labelmap_definition in ['LabelMap', 'labelmap', 'labelMap', 'Labelmap']:
        try:
            return experiment.get_log(labelmap_definition).data
        except picsellia.exceptions.ResourceNotFoundError:
            logging.info(f'{labelmap_definition} not found.')

    logging.error('Labelmap not found')
    return None


# https://github.com/pytorch/vision/issues/4395#issuecomment-1086658634
output_path = os.path.join(os.getcwd(), r"models/retinanet.onnx")

if __name__ == '__main__':
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'

    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])

    by_experiment: bool = os.environ.get('experiment_id', None) is not None

    if by_experiment:
        experiment = client.get_experiment_by_id(id=os.environ['experiment_id'])

        model_artifact = experiment.get_artifact('model-latest')

        confidence_threshold = 0.1
        iou_threshold = 0.3
        max_detections = 5000

    else:
        job_id = os.environ["job_id"]
        logging.info(f"Job ID: {job_id}")
        job = client.get_job_by_id(job_id)
        context = job.sync()
        model_version_id = context['model_version_processing_job']['input_model_version_id']
        model_version = client.get_model_version_by_id(model_version_id)

        parameters = context["parameters"]

        confidence_threshold = parameters.get("confidence_threshold", 0.3)
        iou_threshold = parameters.get("iou_threshold", 0.3)
        max_detections = parameters.get("max_det", 5000)

        # todo verify
        model_artifact = model_version.get_file('model-latest')

        experiment = client.get_experiment_by_id(id=model_version.get_context().experiment_id)

        # download experiment Pytorch model
        model_target_path = os.path.join(os.getcwd(), f'models')

    model_weights_path = download_model_version(model_artifact=model_artifact)

    # parameters
    experiment_parameters = experiment.get_log('All parameters').data

    image_size = experiment_parameters['image_size']
    min_confidence = experiment_parameters['confidence_threshold']
    min_iou_threshold = experiment_parameters['iou_threshold']
    normalization_mean_values = experiment_parameters['augmentations_normalization_mean']
    normalization_std_values = experiment_parameters['augmentations_normalization_std']

    class_mapping = get_model_version_labelmap(experiment)

    # Load retinanet
    # pth_path = "/path/to/retinanet.pth"

    anchor_boxes_params = {
        'sizes': experiment_parameters['anchor_boxes_sizes'],
        'aspect_ratios': experiment_parameters['anchor_boxes_aspect_ratios']
    }

    retinanet = build_retinanet_model(num_classes=len(class_mapping),
                                      use_COCO_pretrained_weights=False,
                                      score_threshold=confidence_threshold,
                                      iou_threshold=iou_threshold,
                                      unfrozen_layers=experiment_parameters['unfreeze'],
                                      mean_values=normalization_mean_values,
                                      std_values=normalization_std_values,
                                      trained_weights=model_weights_path,
                                      anchor_boxes_params=anchor_boxes_params,
                                      image_size=image_size,
                                      max_det=max_detections)
    retinanet.topk_candidates = max_detections*2

    retinanet.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retinanet.to(device)

    # Image sizes
    original_image_size = image_size

    # Batch images hack
    # /!\ torchvision version dependent ???
    retinanet.transform.batch_images = lambda x, size_divisible: x[0].unsqueeze(0)

    dummy_input = torch.randn(1, 3, original_image_size[0], original_image_size[1])
    # dummy_input = preprocess_image(dummy_input)
    image_size = tuple(dummy_input.shape[2:])
    dummy_input = dummy_input.to(device)

    # ONNX export
    torch.onnx.export(
        retinanet,
        dummy_input,
        output_path,
        verbose=True,
        opset_version=13,
        input_names=["images"],
        output_names=['boxes', 'scores', 'labels']
    )

    onnx_graph_name = 'model_latest_onnx'

    logging.info("Try to store model")
    if by_experiment:
        experiment.store(name=onnx_graph_name, path=output_path, do_zip=True)
    else:
        logging.info("Saving model")
        try:
            model_version.store(name=onnx_graph_name, path=output_path, replace=True)
            logging.info("Successfully stored model")
            job.update_job_run_with_status(JobRunStatus.SUCCEEDED)
        except Exception as e:
            print(str(e))
            logging.error(str(e))
            job.update_job_run_with_status(JobRunStatus.FAILED)

    # model_version.store("onnx-model-quantized", model_int8_path)
