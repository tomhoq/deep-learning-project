from sys import stdout
from typing import Literal
import torch
from torch import Tensor
from tqdm import tqdm
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
    - pred_boxes (Tensor): Predicted boxes, shape (N, 7), where columns are [train_idx, class_pred, prob_score, x1, y1, x2, y2].
    - target_boxes (Tensor): Ground truth boxes, shape (M, 7), where columns are [train_idx, class_true, score, x1, y1, x2, y2].
    - box_format (str): "midpoint" or "corners".
    - threshold (float): Minimum DICE score to consider a valid match.

    Returns:
    - mean_dice (float): Mean DICE score after matching.
    """

    if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return 0.0

    # Get unique train_idx values
    train_indices = torch.unique(pred_boxes[:, 0])

    # Initialize list to store DICE scores
    total_dice_scores = []

    tq = tqdm(total=len(train_indices), desc='Calculating DICE score', file=stdout)

    for train_idx in train_indices:
        # Filter boxes for the current train_idx
        pred_group = pred_boxes[pred_boxes[:, 0] == train_idx][:, 3:]  # Extract [x1, y1, x2, y2]
        target_group = target_boxes[target_boxes[:, 0] == train_idx][:, 3:]  # Extract [x1, y1, x2, y2]

        if pred_group.shape[0] == 0 or target_group.shape[0] == 0:
            tq.update()
            continue

        # Perform pairwise DICE score computation (broadcasting)
        pred_group = pred_group.unsqueeze(1)  # Shape (N, 1, 4)
        target_group = target_group.unsqueeze(0)  # Shape (1, M, 4)
        dice_scores = dice(pred_group, target_group, box_format)  # Shape (N, M)

        # Filter matches with DICE scores above the threshold
        valid_dice_scores = dice_scores[dice_scores > threshold]
        total_dice_scores.extend(valid_dice_scores.tolist())

        tq.update()

    tq.close()

    # Compute mean DICE score
    if len(total_dice_scores) == 0:
        return 0.0
    else:
        return np.mean(total_dice_scores)
