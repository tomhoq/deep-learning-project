import numpy as np
import torch
import torch.nn.functional as F
from models.yolo.utils.dice import dice, match_and_calculate_dice
from models.yolo.utils.helpers import get_bboxes
import logging
from mean_average_precision import MetricBuilder


def convert_box_coords(x, y, w, h, class_pred, prob_score):
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    return (xmin, ymin, xmax, ymax, class_pred, prob_score)


def add_metrics(metrics, predictions, ground_truths):
    # Group data by train_idx
    grouped_data = {}

    for entry in ground_truths:
        train_idx, class_pred, prob_score, x, y, w, h = entry

        if train_idx not in grouped_data:
            grouped_data[train_idx] = { 'pred': [], 'gt': [] }

        grouped_data[train_idx]['gt'].append(convert_box_coords(x, y, w, h, class_pred, prob_score))

    for entry in predictions:
        train_idx, class_pred, prob_score, x, y, w, h = entry
        grouped_data[train_idx]['pred'].append(convert_box_coords(x, y, w, h, class_pred, prob_score))

    # Add to metrics
    for train_idx, entries in grouped_data.items():
        gt = np.array([[xmin, ymin, xmax, ymax, class_id, 0, 0] for xmin, ymin, xmax, ymax, class_id, _ in entries['gt']])
        preds = np.array([[xmin, ymin, xmax, ymax, class_id, confidence] for xmin, ymin, xmax, ymax, class_id, confidence in entries['pred']])
        metrics.add(preds, gt)

    return grouped_data




@torch.no_grad()
def validation(model: torch.nn.Module, loss_function, valid_loader, device, scheduler):
    num_classes = 1
    iou_threshold = 0.5

    pred_boxes, target_boxes, valid_loss, inference_time = get_bboxes(
        valid_loader, 
        model, 
        iou_threshold=iou_threshold,
        threshold=0.4,
        S = 7,
        C = num_classes,
        device = device, 
        loss_function = loss_function,
        track_time = True,
    )

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    d = add_metrics(metric_fn, pred_boxes, target_boxes)
    mean_avg_prec = metric_fn.value(iou_thresholds=iou_threshold)['mAP'] 

    dice_score = match_and_calculate_dice(d)

    print('    Valid loss: {:.5f}, mAP: {:.5f}, DICE: {:.5f}, Inference time: {:.2f}ms\n'.format(valid_loss, mean_avg_prec, dice_score, inference_time))

    if scheduler is not None:
        scheduler.step(mean_avg_prec)

    return {
        'valid_loss': valid_loss.item(), 
        'mAP': mean_avg_prec.item(),
        'dice': dice_score.item(),
        'inference_time_ms': inference_time,
    }