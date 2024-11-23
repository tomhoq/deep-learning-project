from typing import Literal
import torch
from torch import Tensor
from models.yolo.utils.intersection_over_union import get_intersection_and_areas
import numpy as np


def dice(boxes_preds: Tensor, boxes_labels: Tensor, box_format: Literal['midpoint', 'corners'] = 'midpoint'):
    """
    Calculates DICE score
    """

    intersection, box1_area, box2_area = get_intersection_and_areas(boxes_preds, boxes_labels, box_format)
    return (2*intersection) / (box1_area + box2_area + 1e-6)


"""
DONT KNOW IF THIS ACTUALLY WORKS OR NOT.
"""
def match_and_calculate_dice(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, box_format: Literal['midpoint', 'corners'] = 'midpoint', threshold: float = 0.5):
    """
    Matches predicted boxes to target boxes and calculates the DICE score.

    Args:
    - pred_boxes (Tensor): Predicted boxes, shape (N, 4).
    - target_boxes (Tensor): Ground truth boxes, shape (M, 4).
    - box_format (str): "midpoint" or "corners".
    - threshold (float): Minimum DICE score to consider a valid match.

    Returns:
    - mean_dice (float): Mean DICE score after matching.
    """

    if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        # Handle edge cases where there are no predictions or targets
        return 0.0

    # Compute DICE scores between all predictions and targets
    dice_scores = []
    for i, pred_box in enumerate(pred_boxes):
        for j, target_box in enumerate(target_boxes):
            score = dice(pred_box.unsqueeze(0), target_box.unsqueeze(0), box_format).item()

            # Skip if score is 0 ?????
            if score > 0:
                dice_scores += [score]

    if len(dice_scores) == 0:
        return 0
    else:
        return np.mean(dice_scores)

