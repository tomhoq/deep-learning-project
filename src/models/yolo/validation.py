import torch
import torch.nn.functional as F
from models.yolo.utils.dice import dice, match_and_calculate_dice
from models.yolo.utils.mean_average_precision import mean_average_precision
from models.yolo.utils.helpers import get_bboxes
import logging


@torch.no_grad()
def validation(model: torch.nn.Module, loss_function, valid_loader, device, scheduler):
    pred_boxes, target_boxes, valid_loss = get_bboxes(
        valid_loader, 
        model, 
        iou_threshold=0.5,
        threshold=0.4,
        S = 7,
        C = 1,
        device = device, 
        loss_function = loss_function
    )
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

    dice_score = match_and_calculate_dice(torch.tensor(pred_boxes), torch.tensor(target_boxes))

    print('    Valid loss: {:.5f}, mAP: {:.5f}, DICE: {:.5f}\n'.format(valid_loss, mean_avg_prec, dice_score))

    scheduler.step(mean_avg_prec)

    return {
        'valid_loss': valid_loss.item(), 
        'mAP': mean_avg_prec.item(),
        'dice': dice_score,
    }