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
def match_and_calculate_dice(grouped_data, box_format: Literal['midpoint', 'corners'] = 'midpoint', threshold: float = 0.5):
    """
    Matches predicted boxes to target boxes and calculates the DICE score.

    Args:
    - grouped_data
    - box_format (str): "midpoint" or "corners".
    - threshold (float): Minimum DICE score to consider a valid match.

    Returns:
    - mean_dice (float): Mean DICE score after matching.
    """

    total_dice_scores = []

    tq = tqdm(total=len(grouped_data), desc='Calculating DICE score', file=stdout)
    for train_idx, entries in grouped_data.items():
        if len(entries['pred']) == 0:
            tq.update()
            continue

        pred_group = torch.tensor(entries['pred'])[:, :4]
        target_group = torch.tensor(entries['gt'])[:, :4]

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

