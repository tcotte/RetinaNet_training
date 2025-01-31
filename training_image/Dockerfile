FROM nvcr.io/nvidia/pytorch:24.01-py3
#FROM  nvcr.io/nvidia/pytorch:22.07-py3
#FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04


COPY training_image/requirements.txt .

RUN pip install -r requirements.txt --trusted-host developer.download.nvidia.com

# https://github.com/albumentations-team/albumentations/issues/1043#issuecomment-2624240007
RUN pip install albumentations==2.0.2
RUN pip uninstall opencv-python opencv-python-headless -y
RUN pip install opencv-python==4.7.0.72
RUN pip install picsellia --upgrade
RUN pip install matplotlib --upgrade
RUN pip install python-dateutil --upgrade
RUN pip install torchmetrics --upgrade


WORKDIR /picsellia_folder

COPY training_image/picsellia_folder .

ENTRYPOINT ["python", "main.py"]
