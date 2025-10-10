import logging
import os
import shutil
import sys
import time
import zipfile
from fnmatch import fnmatch
from typing import Union

from picsellia import Artifact, ModelFile

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def extract_zip_file(zip_file_path: str, destination_folder: str) -> None:
    """
    Extract zip file in destination folder
    :param zip_file_path: path of zip file to extract
    :param destination_folder: destination folder where the zip file has to be extracted
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(destination_folder)


def find_files_in_recursive_dirs(root_directory: str, extension: str) -> list[str]:
    list_files = []

    pattern = f'*.{extension}'

    for path, subdirs, files in os.walk(root_directory):
        for name in files:
            if fnmatch(name, pattern):
                list_files.append(os.path.join(path, name))

    return list_files


def download_model_version(model_artifact: Union[Artifact, ModelFile], model_target_path: str = 'tmp') -> str:
    """
    Download model artifact from Picsellia. Model artifact comes from Picsellia experiment. Model file comes from
    Picsellia model version.
    The model is downloaded as zipfile. This function extracts the zip file and cut/paste the model 'pth' in the
    specified 'model_target_path' folder and removes the folder created by zipfile extraction.
    :param model_target_path: folder where the model will be saved
    :param model_artifact: model artifact from Picsellia experiment
    :return: path of downloaded model
    """
    model_artifact.download(target_path=model_target_path)

    time.sleep(5)

    zip_file_path = os.path.join(model_target_path, os.listdir(model_target_path)[0])
    extract_zip_file(zip_file_path=zip_file_path,
                     destination_folder=model_target_path)

    time.sleep(5)

    pth_file = find_files_in_recursive_dirs(root_directory=model_target_path, extension='pth')[0]
    destination_pth_file = os.path.join(model_target_path, os.path.basename(pth_file))
    os.rename(pth_file, destination_pth_file)

    # clean folder
    subfolders = [f.path for f in os.scandir(model_target_path) if f.is_dir()]
    [shutil.rmtree(dir_) for dir_ in subfolders]
    os.remove(zip_file_path)

    return destination_pth_file