import time
import typing
import os

import cv2
import numpy as np
import torch
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from picsellia import Client
import sys

sys.path.append(r'/training_image/picsellia_folder')

from training_image.picsellia_folder.dataset import PascalVOCDataset
from training_image.picsellia_folder.model_retinanet import collate_fn, build_retinanet_model

'''
We have to get a coherent set of libraries (Pytorch / CUDNN / onnxruntime) to run ONNX with CUDA:
See the link below for more details:
https://github.com/microsoft/onnxruntime/issues/22198#issuecomment-2376010703

Our environment:
CUDA: 12.1
cuDNN: 9.1.0
onnxruntime: 1.19.2

Documentation ONNX runtime on CUDA:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html 
'''
BATCH_SIZE = 1


def get_CUDA_and_CUDNN_versions() -> dict:
    cudnn = torch.backends.cudnn.version()
    cudnn_major = cudnn // 10000
    cudnn = cudnn % 1000
    cudnn_minor = cudnn // 100
    cudnn_patch = cudnn % 100

    return {
        'CUDA': torch.version.cuda,
        'CUDNN': '.'.join([str(cudnn_major), str(cudnn_minor), str(cudnn_patch)])
    }


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ONNX_MODEL_PATH: typing.Final[str] = (
    r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\exports\retinanet.onnx')

if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    # torch.jit.optimized_execution(True)

    os.environ['CUDA_CACHE_DISABLE'] = '0'

    providers = ["CUDAExecutionProvider"]
    # providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options, providers=providers)
    print("Available Providers:", ort.get_available_providers())
    print(sess.get_providers())

    io_binding = sess.io_binding()

    # Pass gpu_graph_id to RunOptions through RunConfigs
    ro = ort.RunOptions()

    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])
    experiment = client.get_experiment_by_id(id=os.environ['experiment_id'])

    # parameters
    experiment_parameters = experiment.get_log('All parameters').data
    image_size = experiment_parameters['image_size']
    min_confidence = experiment_parameters['confidence_threshold']
    min_iou_threshold = experiment_parameters['iou_threshold']
    normalization_mean_values = experiment_parameters['augmentations_normalization_mean']
    normalization_std_values = experiment_parameters['augmentations_normalization_std']

    # load pt model
    model_weights_path = r'/exports/0197a212-cc33-78a1-8d93-ed92b524f42b-model/latest.pth'
    anchor_boxes_params = {
        'sizes': experiment_parameters['anchor_boxes_sizes'],
        'aspect_ratios': experiment_parameters['anchor_boxes_aspect_ratios']
    }

    torch_model = build_retinanet_model(num_classes=len(experiment.get_log('LabelMap').data),
                                        use_COCO_pretrained_weights=False,
                                        score_threshold=0.3,
                                        iou_threshold=0.3,
                                        unfrozen_layers=experiment_parameters['unfreeze'],
                                        mean_values=normalization_mean_values,
                                        std_values=normalization_std_values,
                                        trained_weights=model_weights_path,
                                        anchor_boxes_params=anchor_boxes_params)

    valid_transform = A.Compose([
        A.Resize(*image_size),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    val_dataset = PascalVOCDataset(
        data_folder=r"C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\inferences\dataset\test",
        split='test',
        single_cls=True,
        transform=valid_transform)

    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # collate_fn=train_dataset.collate_fn
        collate_fn=collate_fn
    )

    # compute ONNX Runtime output prediction
    nb_img = 0
    for images, targets in data_loader:
        x = torch.unsqueeze(images[0], dim=0)
        x_copy = torch.tensor(x)

        x = x_copy.cuda()
        torch_model.cuda()
        torch_model.eval()

        start_prediction_torch = time.time()
        with torch.no_grad():
            torch_outputs = torch_model(x)[0]
        time_taken_torch = time.time() - start_prediction_torch
        print(f"Time taken to run torch model: {time_taken_torch}")

        start_prediction_onnx = time.time()
        ort_inputs = {sess.get_inputs()[0].name: to_numpy(x)}
        ort_outs = sess.run(None, ort_inputs)
        time_taken_onnx = time.time() - start_prediction_onnx
        print(f"Time taken to run ONNX: {time_taken_onnx}")

        ort_predictions = np.hstack((ort_outs[0],
                                     np.expand_dims(ort_outs[2], axis=1),
                                     np.expand_dims(ort_outs[1], axis=1)))

        print(ort_predictions.shape)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        image = torch.squeeze(x).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        for i in ort_predictions:
            cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)

        ax[0].imshow(image)





        image = torch.squeeze(x).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        boxes = torch_outputs['boxes'].cpu().numpy().astype(int)
        for i in boxes:
            cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)

        ax[1].imshow(image)

        ax[0].set_title('ONNX inference \n'
                        f'time taken: {time_taken_onnx:.3f}s\n'
                        f'Number of predictions: {len(ort_predictions)}')

        ax[1].set_title('Torch inference \n'
                        f'time taken: {time_taken_torch:.3f}s\n'
                        f'Number of predictions: {len(boxes)}')

        nb_img += 1
        plt.savefig(f'output_inference/inference_{nb_img}.png')
