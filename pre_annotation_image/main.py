import logging
import os.path
from distutils.util import strtobool

import picsellia
import torch.cuda
from picsellia import Client

from annotator import PreAnnotator

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def str2bool(str_value: str) -> bool:
    return str_value.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    use_picsellia_processing: bool = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Using device: {device}")

    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]

    client = Client(api_token=api_token, organization_id=organization_id)

    try:
        job_id = os.environ["job_id"]
        job = client.get_job_by_id(job_id)
        logging.info(f"Job id: {job_id}")

        context = job.sync()["dataset_version_processing_job"]

        input_dataset_version_id = context["input_dataset_version_id"]
        output_dataset_version = context["output_dataset_version_id"]
        model_version_id = context["model_version_id"]

    except picsellia.exceptions.ResourceNotFoundError as e:
        logging.error(f"No job id found: {str(e)}")
        context = {'parameters': {}}

        input_dataset_version_id = '019999e6-726a-7fb5-8687-afe3c1219dcd'
        model_version_id = '0199dc88-523a-7dad-a7db-68f3503f336f'

    except KeyError as e:
        logging.error(f"No job id found: {str(e)}")
        context = {'parameters': {}}

        input_dataset_version_id = os.environ["input_dataset_version_id"]
        model_version_id = os.environ["model_version_id"]
        # output_dataset_version = os.environ["output_dataset_version_id"]

    parameters = context["parameters"]

    confidence_threshold = parameters.get("confidence_threshold", 0.4)
    """image_size = parameters.get("image_size", 1024)
    model_type = parameters.get("model_type", "onnx")"""
    image_size = parameters.get("image_size", 2528)
    model_type = parameters.get("model_type", "torch")
    max_detections = parameters.get("max_det", 5000)

    # if 'single_class' in parameters:
    #     if isinstance(parameters['single_class'], str):
    #         single_class = strtobool(val=parameters['single_class'])
    #     else:
    #         single_class = parameters['single_class']

    logging.info(f"Used parameters:")
    for k, v in parameters.items():
        logging.info(f"{k}: {v}")

    pre_annotator = PreAnnotator(client=client,
                                 dataset_version_id=input_dataset_version_id,
                                 model_version_id=model_version_id,
                                 parameters=parameters,
                                 img_size=image_size,
                                 model_type=model_type,
                                 max_detections_per_image=max_detections)

    pre_annotator.setup_pre_annotation_job()
    pre_annotator.pre_annotate(confidence_threshold)

    logging.info("Pre-annotation done!")
