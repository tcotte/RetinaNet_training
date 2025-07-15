import yaml

from training_image.picsellia_folder.utils import read_yaml_file

def set_nested_value(d, keys, value):
    """Sets a value in a nested dictionary using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})  # Navigate or create intermediate dictionaries
    d[keys[-1]] = value

def write_yaml_file(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

if __name__ == '__main__':
    config_dict = read_yaml_file(file_path=r'config_files/config.yaml')
    # config_dict['backbone']['backbone_layers_nb'] = 34
    # new_setup = {
    #     'backbone': {
    #         'backbone_layers_nb': 34
    #     }
    # }
    set_nested_value(config_dict, ['backbone', 'backbone_layers_nb'], 156)

    write_yaml_file(data=config_dict, file_path=r'config_files/config_temp.yaml')