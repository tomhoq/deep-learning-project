from typing import Literal
import torch
from torch import Tensor
from models.yolo.utils.intersection_over_union import get_intersection_and_areas


def dice(boxes_preds: Tensor, boxes_labels: Tensor, box_format: Literal['midpoint', 'corners'] = 'midpoint'):
    """
    Calculates DICE score
    """

    intersection, box1_area, box2_area = get_intersection_and_areas(boxes_preds, boxes_labels, box_format)
    return (2*intersection) / (box1_area + box2_area + 1e-6)

