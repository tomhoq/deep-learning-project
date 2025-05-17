from sys import stdout
from typing import Literal
import torch
from torch import Tensor
from tqdm import tqdm
from models.yolo.utils.intersection_over_union import get_intersection_and_areas
import numpy as np
from PIL import Image, ImageDraw
import cv2


def dice_score(mask1, mask2):
    # Ensure the masks are binary
    mask1 = ~mask1.astype(bool)
    mask2 = ~mask2.astype(bool)
    
    # Compute intersection and union
    intersection = np.sum(mask1 & mask2)
    total_pixels = np.sum(mask1) + np.sum(mask2)
    
    # Handle division by zero
    if total_pixels == 0:
        return 1.0 if np.sum(mask1 == mask2) == mask1.size else 0.0
    
    # Compute Dice Score
    dice = 2 * intersection / total_pixels
    return dice


def draw_bboxes_on_mask(boxes, image_size):
    """
    Draw bounding boxes on a binary mask.
    """
    mask = Image.new('1', image_size, 0)  # Binary mask
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()

        x1  = int(x1 * image_size[0])
        y1  = int(y1 * image_size[0])
        x2  = int(x2 * image_size[0])
        y2  = int(y2 * image_size[0])

        draw.rectangle([x1, y1, x2, y2], outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)  # Convert to numpy array


def dice(boxes_preds: Tensor, boxes_labels: Tensor, box_format: Literal['midpoint', 'corners'] = 'midpoint'):
    """
    Calculates DICE score
    """

    intersection, box1_area, box2_area = get_intersection_and_areas(boxes_preds, boxes_labels, box_format)
    return (2*intersection) / (box1_area + box2_area + 1e-6)


"""
DONT KNOW IF THIS ACTUALLY WORKS OR NOT.
"""
def calculate_dice_yolo(grouped_data, batch_size, img_size = (448, 448), threshold: float = 0.5):
    dices = []

    tq = tqdm(total=len(grouped_data), desc='Calculating DICE score', file=stdout)
    for train_idx, entries in grouped_data.items():
        if len(entries['pred']) == 0:
            tq.update()
            continue

        pred_group = torch.tensor(entries['pred'])
        target_group = torch.tensor(entries['gt'])

        pred_boxes = pred_group[..., 0:4]  # Bounding boxes [x1, y1, x2, y2]
        pred_scores = pred_group[..., 5]  # Objectness scores
        pred_boxes = pred_boxes[pred_scores > threshold]

        target_boxes = target_group[..., 0:4]  # Bounding boxes [x1, y1, x2, y2]
        target_scores = target_group[..., 5]  # Objectness scores
        target_boxes = target_boxes[target_scores > threshold]

        pred = torch.tensor([])
        if len(pred_boxes) > 0:
            pred = torch.tensor(draw_bboxes_on_mask(pred_boxes, img_size))

        true = torch.tensor([])
        if len(target_boxes) > 0:
            true = torch.tensor(draw_bboxes_on_mask(target_boxes, img_size))

        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            dice = dice_score(pred.numpy(), true.numpy())
            dices.append(dice)
        elif len(target_boxes) > 0:
            dices.append(0)
        
        tq.update()

    tq.close()

    dice = torch.tensor(dices).mean()

    return dice
