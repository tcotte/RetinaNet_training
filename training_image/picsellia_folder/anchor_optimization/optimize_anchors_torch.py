import warnings
from typing import Tuple

import pandas as pd
from pandas import read_csv
from torchvision.models.detection.anchor_utils import AnchorGenerator

try:
    from .utils_anchor_boxes.compute_overlap import compute_overlap
except ImportError:
    from utils_anchor_boxes.compute_overlap import compute_overlap

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys
import numpy as np
import scipy.optimize
from PIL import Image

import sys

import xml.etree.ElementTree as ET

# global variable
global state
state = {'best_result': sys.maxsize}


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def compute_anchor_sizes(scales, sizes):
    formats_ = []
    for size in sizes:
        formats_.append(tuple([int(np.round(r * size)) for r in scales]))
    return formats_


def calculate_config(values,
                     ratio_count,
                     SIZES=[32, 64, 128, 256, 512],
                     STRIDES=[8, 16, 32, 64, 128]):
    split_point = int((ratio_count - 1) / 2)

    ratios = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1 / values[i])

    scales = values[split_point:]

    return AnchorParameters(SIZES, STRIDES, ratios, scales)


def base_anchors_for_shape(pyramid_levels=None,
                           anchor_params=None):
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchor_generator = AnchorGenerator()
        anchors = anchor_generator.generate_anchors(
            scales=[scale * anchor_params.sizes[idx] for scale in anchor_params.scales],
            aspect_ratios=anchor_params.ratios)
        # anchors = keras_retinanet.utils.anchors.generate_anchors(
        #     base_size=anchor_params.sizes[idx],
        #     ratios=anchor_params.ratios,
        #     scales=anchor_params.scales
        # )
        all_anchors = np.append(all_anchors, anchors, axis=0)

    return all_anchors


def average_overlap(values,
                    entries,
                    image_shape,
                    mode='focal',
                    ratio_count=3,
                    include_stride=False,
                    SIZES=[32, 64, 128, 256, 512],
                    STRIDES=[8, 16, 32, 64, 128],
                    verbose=False,
                    set_state=None,
                    to_tuple=False,
                    threads=1):
    anchor_params = calculate_config(values,
                                     ratio_count,
                                     SIZES,
                                     STRIDES)
    # if include_stride:
    #     anchors = anchors_for_shape(image_shape, anchor_params=anchor_params)
    # else:
    anchors = base_anchors_for_shape(anchor_params=anchor_params)

    overlap = compute_overlap(entries, anchors)
    max_overlap = np.amax(overlap, axis=1)
    not_matched = len(np.where(max_overlap < 0.5)[0])

    if mode == 'avg':
        result = 1 - np.average(max_overlap)
    elif mode == 'ce':
        result = np.average(-np.log(max_overlap))
    elif mode == 'focal':
        result = np.average(-(1 - max_overlap) ** 2 * np.log(max_overlap))
    else:
        raise Exception('Invalid mode.')

    if set_state is not None:
        state = set_state

    # --------------------------------------------------------------------------------------------------------------------------------
    # "scipy.optimize.differential_evolution" utilizes multiprocessing but internally uses "multiprocessing.Pool" and not
    # "multiprocessing.Process" which is required for sharing state between processes
    # (see: https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes)
    #
    # the "state" variable does not affect directly the "scipy.optimize.differential_evolution" process, therefore updates will be
    # printed out in case of improvement only if a single thread is used
    # --------------------------------------------------------------------------------------------------------------------------------

    if threads == 1:

        if result < state['best_result']:
            state['best_result'] = result

            if verbose:
                print('Current best anchor configuration')
                print('State: {}'.format(np.round(state['best_result'], 5)))
                print(
                    'Ratios: {}'.format(
                        sorted(
                            np.round(
                                anchor_params.ratios,
                                3))))
                print(
                    'Scales: {}'.format(
                        sorted(
                            np.round(
                                anchor_params.scales,
                                3))))

            if include_stride:
                if verbose:
                    print(
                        'Average overlap: {}'.format(
                            np.round(
                                np.average(max_overlap),
                                3)))

            if verbose:
                print(
                    "Number of labels that don't have any matching anchor: {}".format(not_matched))
                print()

    if to_tuple:
        # return a tuple, which happens in the last call to the 'average_overlap' function
        return result, not_matched
    else:
        return result


def anchors_optimize(annotations,
                     ratios=3,
                     scales=3,
                     objective='focal',
                     popsize=15,
                     mutation=0.5,
                     image_min_side=800,
                     image_max_side=1333,
                     # default SIZES values
                     SIZES=[32, 64, 128, 256, 512],
                     # default STRIDES values
                     STRIDES=[8, 16, 32, 64, 128],
                     include_stride=False,
                     resize=False,
                     threads=1,
                     verbose=False,
                     seed=None):
    """
    Important Note: The python "anchors_optimize" function is meant to be used from the command line (from within a Python console it gives incorrect results)    
    """

    if ratios % 2 != 1:
        raise Exception('The number of ratios has to be odd.')

    entries = np.zeros((0, 4))
    max_x = 0
    max_y = 0

    updating = 'immediate'
    if threads > 1:
        # when the number of threads is > 1 then 'updating' is set to 'deferred' by default (see the documentation of "scipy.optimize.differential_evolution())
        updating = 'deferred'

    if seed is None:
        seed = np.random.RandomState()
    else:
        seed = np.random.RandomState(seed)

    if verbose:
        print('Loading object dimensions.')

    df = read_csv(annotations, header=None)
    for line, row in df.iterrows():
        x1, y1, x2, y2 = list(map(lambda x: int(x), row[1:5]))

        if not x1 or not y1 or not x2 or not y2:
            continue

        if resize:
            # Concat base path from annotations file follow retinanet
            base_dir = os.path.split(annotations)[0]
            relative_path = row[0]
            image_path = os.path.join(base_dir, relative_path)
            img = Image.open(image_path)

            if hasattr(img, "shape"):
                image_shape = img.shape
            else:
                image_shape = (img.size[0], img.size[1], 3)

            scale = compute_resize_scale(
                image_shape, min_side=image_min_side, max_side=image_max_side)
            x1, y1, x2, y2 = list(map(lambda x: int(x) * scale, row[1:5]))

        max_x = max(x2, max_x)
        max_y = max(y2, max_y)

        if include_stride:
            entry = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
            entries = np.append(entries, entry, axis=0)
        else:
            width = x2 - x1
            height = y2 - y1
            entry = np.expand_dims(
                np.array([-width / 2, -height / 2, width / 2, height / 2]), axis=0)
            entries = np.append(entries, entry, axis=0)

    image_shape = [max_y, max_x]

    if verbose:
        print('Optimising anchors.')

    bounds = []

    for i in range(int((ratios - 1) / 2)):
        bounds.append((1, 4))

    for i in range(scales):
        bounds.append((0.4, 2))

    update_state = None
    if threads == 1:
        update_state = state

    ARGS = (entries,
            image_shape,
            objective,
            ratios,
            include_stride,
            SIZES,
            STRIDES,
            verbose,
            update_state,
            # return a single value ('to_tuple' parameter is set to False)
            False,
            threads)

    result = scipy.optimize.differential_evolution(func=average_overlap,
                                                   # pass the '*args' as a tuple (see: https://stackoverflow.com/q/32302654)
                                                   args=ARGS,
                                                   mutation=mutation,
                                                   updating=updating,
                                                   workers=threads,
                                                   bounds=bounds,
                                                   popsize=popsize,
                                                   seed=seed)

    if hasattr(result, 'success') and result.success:
        print('Optimization ended successfully!')
    elif not hasattr(result, 'success'):
        print('Optimization ended!')
    else:
        print('Optimization ended unsuccessfully!')
        print('Reason: {}'.format(result.message))

    values = result.x
    anchor_params = calculate_config(values,
                                     ratios,
                                     SIZES,
                                     STRIDES)

    (avg, not_matched) = average_overlap(values,
                                         entries,
                                         image_shape,
                                         'avg',
                                         ratios,
                                         include_stride,
                                         SIZES,
                                         STRIDES,
                                         verbose,
                                         # pass a specific value to the 'set_state' parameter
                                         {'best_result': 0},
                                         # return a 'tuple'  ('to_tuple' parameter is set to True)
                                         True,
                                         # set the 'threads' parameter to 1
                                         1)

    # as 'end_state' set the 'avg' value
    end_state = np.round(avg, 5)
    RATIOS_result = sorted(np.round(anchor_params.ratios, 3))
    SCALES_result = sorted(np.round(anchor_params.scales, 3))

    anchor_sizes = compute_anchor_sizes(scales=SCALES_result, sizes=SIZES)

    print()
    print('Final best anchor configuration')
    print('State: {}'.format(end_state))
    print('Ratios: {}'.format(RATIOS_result))
    print('Scales: {}'.format(SCALES_result))
    print('Anchor sizes: {}'.format(anchor_sizes))

    dict_out = {
        'ratios': RATIOS_result,
        'sizes': anchor_sizes,
        'not_matched': not_matched,
        'end_state': end_state}

    if include_stride:
        STRIDE = np.round(1 - avg, 3)
        print('Average overlap: {}'.format(STRIDE))
        dict_out['stride'] = STRIDE

    print("Number of labels that don't have any matching anchor: {}".format(not_matched))

    return dict_out


def parse_annotations(xml_file: str):
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

    # if not single_cls:
    #     class_ids = [
    #         list(self.class_mapping.keys())[list(self.class_mapping.values()).index(cls)]
    #         for cls in classes
    #     ]
    # else:
    #     if self._add_bckd_as_class:
    #         class_ids = [1] * len(boxes)
    #     else:
    #         class_ids = [0] * len(boxes)
    #
    # boxes, class_ids = self.remove_duplicate_labels(boxes=boxes, labels=class_ids)

    return {'boxes': boxes, 'image': image_name}


def compute_optimized_anchors(annotations_path: str, image_size: Tuple[int, int],
                              temp_csv_filepath: str = 'labels.csv') -> dict:
    image_path = os.path.join(os.path.dirname(annotations_path), 'JPEGImages')

    df = pd.DataFrame({
        'filename': [],
        'x0': [],
        'y0': [],
        'x1': [],
        'y1': []
    })

    for index, annotation_file in enumerate(os.listdir(annotations_path)):
        full_annotation_path = os.path.join(annotations_path, annotation_file)
        dict_ = parse_annotations(full_annotation_path)

        if not len(np.array(dict_['boxes'])) == 0:
            df_temp = pd.DataFrame({
                'filename': [os.path.join(image_path, dict_['image'])] * len(dict_['boxes']),
                'x0': np.array(dict_['boxes'])[:, 0],
                'y0': np.array(dict_['boxes'])[:, 1],
                'x1': np.array(dict_['boxes'])[:, 2],
                'y1': np.array(dict_['boxes'])[:, 3]
            })
            df = pd.concat([df, df_temp])

    df.to_csv(temp_csv_filepath, header=None, index=False)

    optimized_anchor_boxes_params = anchors_optimize(annotations=temp_csv_filepath,
                                                     image_min_side=min(image_size),
                                                     image_max_side=max(image_size),
                                                     resize=True)
    os.remove(temp_csv_filepath)
    return optimized_anchor_boxes_params

