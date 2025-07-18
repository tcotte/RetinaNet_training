import os

import imutils.paths
import torch
from picsellia import Client
from torchmetrics.detection import MeanAveragePrecision
import sys

from inferences.detect import download_dataset_version
from ultralytics import YOLO
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), r'training_image\picsellia_folder'))

import xml.etree.ElementTree as ET

def parse_annotation(xml_file, single_cls: bool = False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        if xmin >= xmax:
            # print(f"Error loading bounding box in {image_name}")
            pass
        elif ymin >= ymax:
            pass
            # print(f"Error loading bounding box in {image_name}")
        else:
            boxes.append([xmin, ymin, xmax, ymax])

    return {'boxes': boxes, 'labels': [0]*len(boxes)}

if __name__ == '__main__':
    dataset_root_folder = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\dataset'
    # model_weights_path = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\models\latest.pth'

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)
    # Get experiment
    experiment = client.get_experiment_by_id(id=os.environ["experiment_id"])


    # model_target_path: str = 'tmp/'
    # shutil.rmtree(model_target_path, ignore_errors=True)
    # experiment.get_artifact(name='pretrained-weights').download(target_path=model_target_path)

    model_weights_path = r'models/YOLO/yolov8-best.onnx'
    # model_weights_path = 'tmp/latest.pth'

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Download datasets
    datasets = experiment.list_attached_dataset_versions()

    if not os.path.exists(dataset_root_folder):
        download_dataset_version(root=dataset_root_folder, alias='test', experiment=experiment)

    model = YOLO(model_weights_path)

    metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True, class_metrics=True,
                                  max_detection_thresholds=[3000, 5000, 10000])

    for image in list(imutils.paths.list_images(os.path.join(dataset_root_folder, 'test'))):
        prediction = model.predict(image, imgsz=1024)

        annotation_file = os.path.join(dataset_root_folder, 'test', 'Annotations',
                                       os.path.basename(image).replace('.jpg', '.xml'))
        target = parse_annotation(xml_file=annotation_file)

        predictions = [{'boxes': prediction[0].boxes.xyxy, 'labels': torch.zeros(len(prediction[0].boxes.cls)),
                        'scores': prediction[0].boxes.conf}]

        targets = [{k: torch.Tensor(v) for k, v in target.items()}]

        targets[0]['labels'] = targets[0]['labels'].int()
        predictions[0]['labels'] = predictions[0]['labels'].int()

        metric.update(predictions, targets)

    validation_metrics = metric.compute()
    print(f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
          f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
          f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
          f"'Precision': {float(validation_metrics['precision'][0][25][0][0][-1]):.3} / "
          f"'Recall': {float(validation_metrics['recall'][0][0][0][-1]):.3} ")





