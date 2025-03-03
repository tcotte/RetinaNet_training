import logging
import os
from typing import Optional

import imutils
import imutils.paths
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


def compute_image_parameters(image_path: str) -> dict:
    d = {'r': {}, 'g': {}, 'b': {}}

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    for index, c in enumerate(d.keys()):
        c_channel = img_array[:, :, index].reshape(-1)
        c_std = np.std(c_channel)
        c_mean = np.mean(c_channel)
        c_max = np.max(c_channel)

        d[c] = {
            'std': c_std,
            'mean': c_mean,
            'max': c_max
        }

    return d


def compute_auto_normalization_parameters(image_folder: str, max_workers: Optional[int] = None) -> dict:
    logging.info("Computing auto-normalization parameters...")
    d = {'r': {}, 'g': {}, 'b': {}}

    parallel = Parallel(n_jobs=os.cpu_count() if max_workers is None else max_workers, return_as="generator")
    image_parameters = parallel(delayed(compute_image_parameters)(image_path) for image_path in
                                list(imutils.paths.list_images(image_folder)))

    for index_image, image_parameter in tqdm(enumerate(image_parameters)):
        for index, c in enumerate(d.keys()):
            if index_image == 0:
                    d[c] = image_parameter[c]

            else:
                for k, v in image_parameter[c].items():
                    if k != 'max':
                        d[c][k] += image_parameter[c][k]
                    else:
                        d[c][k] = max(d[c][k], image_parameter[c][k])

    # get max pixel value of all channels
    max_pixel_value = 0
    for c in d.keys():
        if int(d[c]['max']) > max_pixel_value:
            max_pixel_value = int(d[c]['max'])

    list_rgb_mean = []
    list_rgb_std = []

    for c in d.keys():
        for values in ['mean', 'std']:
            d[c][values] /= len(list(imutils.paths.list_images(image_folder)))
            d[c][values] /= max_pixel_value

            if values == 'mean':
                list_rgb_mean.append(d[c][values])

            if values == 'std':
                list_rgb_std.append(d[c][values])

    logging.info(f'Computed normalization parameters: \nMax pixel value {max_pixel_value} \nRGB mean {list_rgb_mean} \n'
                 f'RGB std {list_rgb_std}')

    return {
        'max_pixel_value': max_pixel_value,
        'RGB mean': list_rgb_mean,
        'RGB std': list_rgb_std
    }


# if __name__ == '__main__':
#     image_folder = r'C:\Users\tristan_cotte\PycharmProjects\RetinaNet_training\datasets\train'
#     print(compute_auto_normalization_parameters(image_folder))
