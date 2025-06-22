"""This module contains functions to download and process the mnist dataset.
code modified from: https://github.com/LukeTonin/simple-deep-learning
"""
from typing import List
from typing import Union
import numpy as np
import gc
import random 
from PIL import Image


def overlay_at_random(array1: np.ndarray, array2: np.ndarray,
                      bounding_boxes: List[dict] = None,
                      max_iou: float = 0.2, min_iou: float = 0.1,
                      random_location: bool =True):
    """Overlay an array over another.

    Overlays array2 over array1 while attempting to avoid locations specified by 
    a list of bounding_boxes. This function overlays inplace so array1 is not
    copied or returned.

    THe location of the array2 in array1 is determined at random.

    Parameters:
        array1: The base array (or canvas) on which to overlay array2.
        array2: The second array to overlay over array1.
        max_array_value: The maximum allowed value for this array.
            Any number larger than this will be clipped.
            Clipping is necessary because the overlaying is done by summing arrays.
        bounding_boxes: A list of bounding boxes in the format xyxy.
           The algorithm will not overlay with existing bounding boxes by more
           than an IOU of max_iou.
        max_iou: The maximum allowed IOU between the candidate location and the
            bounding_boxes.

    Returns:
        The bounding box of the added image if successfully overlaid. Otherwise None.
    """
    if not bounding_boxes:
        bounding_boxes = []
        

    height1, width1, *_ = array1.shape
    height2, width2, *_ = array2.shape
    
    
    # maximum x and y
    max_x = width1 - width2
    max_y = height1 - height2
    
    is_valid = False
    # This number is arbitrary. There are better ways of doing this but this is fast enough.
    
    max_attempts = 1000
    attempt = 0
    
    while not is_valid:
        if attempt > max_attempts:
            return None
        else:
            attempt += 1
            
        
        if random_location == False: 
            # objects are centered at the base array
            x= int(max_x/2)
            y= int(max_y/2)
        else:
            x = np.random.randint(max_x + 1)
            y = np.random.randint(max_y + 1)
        

        candidate_bounding_box = {
            'xmin': x,
            'ymin': y,
            'xmax': x + width2,
            'ymax': y + height2,
        }

        is_valid = True
        for bounding_box in bounding_boxes:
            if calculate_iou(bounding_box, candidate_bounding_box) > max_iou or calculate_iou(bounding_box, candidate_bounding_box) < min_iou:
                is_valid = False
                break

    overlay = overlay_array(array1=array1, array2=array2, x=x, y=y)

    
    return overlay, candidate_bounding_box


def overlay_array(array1: np.ndarray, array2: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlay an array on another at a given position.

    Parameters:
        array1: The base array (or canvas) on which to overlay array2.
        array2: The second array to overlay over array1.
        max_array_value: The maximum allowed value for this array.
            Any number larger than this will be clipped.
            Clipping is necessary because the overlaying is done by summing arrays.
    
    Returns:
        array1: array1 with array2 overlaid at the position x, y.

    """

    height1, width1, *other1 = array1.shape
    height2, width2, *other2 = array2.shape

    if height2 > height1 or width2 > width1:
        raise ValueError('array2 must have a smaller shape than array1')

    if other1 != other2:
        raise ValueError('array1 and array2 must have same dimensions beyond dimension 2.')

    array1[y:y+height2, x:x+width2, ...] = array2


    return array1




def format_bounding_box(bounding_box: Union[tuple, dict, np.ndarray, list],
                        input_format: str = None,
                        output_format: str = 'xyxy',
                        output_type: str = 'dict') -> Union[dict, tuple]:
    """Format a bounding box object.

    This is a utility function for converting bounding boxes between different formats.
    There are two caracteristics for a bounding box:
    - format: Whether the bounding box is defined by its min and max values: xmin, ymin, xmax, ymax
        or by its minimum x and y and a width and height.
    - type: Whether or not the bounding box is an indexable (e.g tuple, array, list) or a dictionary.

    This function converts between all types.

    In the case of output_type == 'dict', the keys of the dictionary will be
    reordered and renamed to be either:
    - xmin. ymin, xmax, ymax for the format xyxy.
    - x, y, width, height for the format xywh.

    Parameters:
        bounding_box: The input bounding box, as a tuple, array, list or dictionary.
        input_format: The format of the input. Required if the input type is tuple.
            Otherwise the input format is inferred from the keys of the dictionary.
        output_format: Determines the output format of the bounding box.
            Must be 'xyxy' or 'xywh'. Defaults to 'xyxy'
        output_type: The output type of the bounding box.
            Must be 'dict' or 'tuple'. Defaults to 'dict'.

    Returns:
        return_value: A bounding boxes represented in the specified format and type.
    """
    if output_format == 'xyxy':
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                return_value = {
                    'xmin': bounding_box['xmin'],
                    'ymin': bounding_box['ymin'],
                    'xmax': bounding_box['xmax'],
                    'ymax': bounding_box['ymax']
                }
            elif all(key in bounding_box for key in ['xmin', 'ymin', 'width', 'height']):
                return_value = {
                    'xmin': bounding_box['xmin'],
                    'ymin': bounding_box['ymin'],
                    'xmax': bounding_box['xmin'] + bounding_box['width'],
                    'ymax': bounding_box['ymin'] + bounding_box['height']
                }
            elif all(key in bounding_box for key in ['x', 'y', 'width', 'height']):
                return_value = {
                    'xmin': bounding_box['x'],
                    'ymin': bounding_box['y'],
                    'xmax': bounding_box['x'] + bounding_box['width'],
                    'ymax': bounding_box['y'] + bounding_box['height']
                }
            else:
                raise ValueError(
                    f'Incorrect format for bounding_box dictionary. Received: {bounding_box}')
        else:
            if input_format == 'xyxy':
                return_value = {
                    'xmin': bounding_box[0],
                    'ymin': bounding_box[1],
                    'xmax': bounding_box[2],
                    'ymax': bounding_box[3]
                }
            elif input_format == 'xywh':
                return_value = {
                    'xmin': bounding_box[0],
                    'ymin': bounding_box[1],
                    'xmax': bounding_box[0] + bounding_box[2],
                    'ymax': bounding_box[1] + bounding_box[3]
                }
            else:
                raise ValueError(
                    'If bounding_box is not a dictionary, input_format must be specified: "xyxy" or "xywh"')

    elif output_format == 'xywh':
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ['xmin', 'ymin', 'width', 'height']):
                return_value = {
                    'x': bounding_box['xmin'],
                    'y': bounding_box['ymin'],
                    'width': bounding_box['width'],
                    'height': bounding_box['height']
                }
            elif all(key in bounding_box for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                return_value = {
                    'x': bounding_box['xmin'],
                    'y': bounding_box['ymin'],
                    'width': bounding_box['xmax'] - bounding_box['xmin'],
                    'height': bounding_box['ymax'] - bounding_box['ymin']
                }
            elif all(key in bounding_box for key in ['x', 'y', 'width', 'height']):
                return_value = {
                    'x': bounding_box['x'],
                    'y': bounding_box['y'],
                    'width': bounding_box['width'],
                    'height': bounding_box['height']
                }
            else:
                raise ValueError(
                    f'Incorrect format for bounding_box dictionary. Received: {bounding_box}')
        else:
            if input_format == 'xyxy':
                return_value = {
                    'x': bounding_box[0],
                    'y': bounding_box[1],
                    'width': bounding_box[2] - bounding_box[0],
                    'height': bounding_box[3] - bounding_box[1]
                }
            elif input_format == 'xywh':
                return_value = {
                    'x': bounding_box[0],
                    'y': bounding_box[1],
                    'width': bounding_box[2],
                    'height': bounding_box[3]
                }
            else:
                raise ValueError(
                    'If bounding_box is not a dictionary, input_format must be specified: "xyxy" or "xywh"')
    else:
        raise ValueError(
            f'output_format must be either "xyxy" or "xywh". Received {output_format}')

    if output_type == 'tuple':
        return tuple(return_value.values())
    elif output_type == 'dict':
        return return_value
    else:
        raise ValueError(
            f'output_type must be either "dict" or "tuple". Received {output_type}')


def calculate_iou(bounding_box1: dict, bounding_box2: dict) -> float:
    """Calculate the intersection over union of two bounding boxes.

    Both bounding boxes must be in xyxy format and of type dict.
    See format_bounding_box function for more details.

    Returns:
        IOU: number between 0 and 1.
    """

    A1 = ((bounding_box1['xmax'] - bounding_box1['xmin'])
          * (bounding_box1['ymax'] - bounding_box1['ymin']))
    A2 = ((bounding_box2['xmax'] - bounding_box2['xmin'])
          * (bounding_box2['ymax'] - bounding_box2['ymin']))

    xmin = max(bounding_box1['xmin'], bounding_box2['xmin'])
    ymin = max(bounding_box1['ymin'], bounding_box2['ymin'])
    xmax = min(bounding_box1['xmax'], bounding_box2['xmax'])
    ymax = min(bounding_box1['ymax'], bounding_box2['ymax'])
    
    A_inter = (xmax-xmin) * (ymax - ymin)

    if ymin >= ymax or xmin >= xmax:
        return 0

    return (A_inter) / (A1 + A2 - A_inter)
