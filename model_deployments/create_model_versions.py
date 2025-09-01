"""
The aim of this script is to deploy several ResNet model versions to Picsell.ia: one for each existing architecture.
Each architecture (number of layers) is described in Docker environment variables.
"""
import os
from typing import Optional

import picsellia
import yaml
from picsellia.exceptions import ResourceConflictError
from picsellia import Client, ModelVersion
from picsellia.types.enums import Framework, InferenceType

from training_image.picsellia_folder.utils import read_yaml_file


def set_nested_value(d, keys, value):
    """Sets a value in a nested dictionary using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})  # Navigate or create intermediate dictionaries
    d[keys[-1]] = value


def write_yaml_file(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def create_retinanet_model_version(model: picsellia.Model, nb_layers: int, base_parameters: dict,
                                   name: Optional[str]=None) -> ModelVersion:
    if name is None:
        name = 'RetinaNet' + str(nb_layers)

    model_version = model.create_version(base_parameters=base_parameters,
                         docker_env_variables={'architecture': nb_layers},
                         name=name,
                         type=InferenceType.OBJECT_DETECTION,
                         framework=Framework.PYTORCH,
                         docker_image_name="9d8xtfjr.c1.gra9.container-registry.ovh.net/picsellia/retinanet_training",
                         docker_tag='1.0',
                         docker_flags=['--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864'])
    return model_version


if __name__ == '__main__':

    client = Client(api_token=os.environ['api_token'], organization_id=os.environ['organization_id'])
    model = client.get_model(name='RetinaNet')

    base_parameters = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "lr_scheduler_gamma": 0.9,
        "epoch": 300,
        "patience": 50,
        "optimizer": "Adam",
        "num_workers": 8,
        "coco_pretrained_weights": "True",
        "image_size": 1024,
        "mosaic": 0,
        "cutmix": 0,
        "cutout": 0,
        "mixup": 0
    }

    # ResNet backbone
    for nb_layers in [18, 34, 50, 101, 152]:
        temp_config_filepath = r'config_files/config_temp.yaml'
        config_dict = read_yaml_file(file_path=r'config_files/config.yaml')
        set_nested_value(config_dict, ['backbone', 'backbone_layers_nb'], nb_layers)
        write_yaml_file(data=config_dict, file_path=temp_config_filepath)

        # base_parameters['nb_layers'] = nb_layers
        try:
            model_version = create_retinanet_model_version(model=model, nb_layers=nb_layers, base_parameters=base_parameters)

        # if model version already exists: delete it and recreate it
        except ResourceConflictError:
            model_version = model.get_version(version='RetinaNet' + str(nb_layers))
            model_version.delete()

            model_version = create_retinanet_model_version(model=model, nb_layers=nb_layers, base_parameters=base_parameters)

        finally:
            model_version.store(name='config', path=temp_config_filepath)
            model_version.store(name='aug_hyp', path=r'config_files/aug_hyp.yaml')
            os.remove(temp_config_filepath)

    # ResNeXt backbone
    for nb_layers in [50, 101]:
        temp_config_filepath = r'config_files/config_temp.yaml'
        config_dict = read_yaml_file(file_path=r'config_files/config.yaml')
        set_nested_value(config_dict, ['backbone', 'backbone_layers_nb'], nb_layers)
        set_nested_value(config_dict, ['backbone', 'backbone_type'], 'ResNeXt')
        write_yaml_file(data=config_dict, file_path=temp_config_filepath)

        # base_parameters['nb_layers'] = nb_layers
        try:
            model_version = create_retinanet_model_version(model=model,
                                                           nb_layers=nb_layers,
                                                           base_parameters=base_parameters,
                                                           name='RetinaNet_ResNeXt' + str(nb_layers))

        # if model version already exists: delete it and recreate it
        except ResourceConflictError:
            model_version = model.get_version(version='RetinaNet_ResNeXt' + str(nb_layers))
            model_version.delete()

            model_version = create_retinanet_model_version(model=model,
                                                           nb_layers=nb_layers,
                                                           base_parameters=base_parameters,
                                                           name='RetinaNet_ResNeXt' + str(nb_layers))

        finally:
            model_version.store(name='config', path=temp_config_filepath)
            model_version.store(name='aug_hyp', path=r'config_files/aug_hyp.yaml')
            os.remove(temp_config_filepath)
