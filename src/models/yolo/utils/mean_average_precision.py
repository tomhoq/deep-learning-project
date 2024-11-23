from collections import Counter
from typing import Literal
import torch
from models.yolo.utils import intersection_over_union


"""
Calculates Mean Average Precision (mAP) for object detection.

1. Iterates over all classes.
2. For each class:
   - Collects predictions and ground truths.
   - Matches predictions to ground truths based on IoU.
   - Computes True Positives (TP) and False Positives (FP).
3. Calculates precision-recall curve and integrates to find Average Precision (AP).
4. Returns mAP (mean of APs across all classes).
"""


def mean_average_precision(
    pred_boxes: list, true_boxes: list, iou_threshold: float = 0.5, box_format: Literal['midpoint', 'corners'] = 'midpoint', num_classes: int = 20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key,val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)


        # Sort by box probabilities which is index 2 (higher probability first)
        detections.sort(key=lambda x: x[2], reverse=True)

        # Initialize variables
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        
        # If no true bboxes exists for this class then we can safely skip
        total_true_bboxes = len(ground_truths)
        if total_true_bboxes == 0:
            continue


        for detection_idx, detection in enumerate(detections):

            # Only take out the ground_truths that have the same training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1


        # Calculates precision-recall
        #   - Recall tells us how much of the ground truth was correctly detected --> "Out of all actual objects, how many did we detect?"
        #   - Precision tells us how many of the predicted boxes are correct --> "Out of all the boxes the model predicted, how many were correct?"
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # Add boundary Values to make the precision-recall curve start at (0,1): 
        #   - precision starts at 1 1 when recall is 0 0 (before any detections). 
        #   - recall starts at 0 0 with no detections. 
        # This ensures that the curve covers the full range.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Calculate area under the curve, which is by definition the average precision (torch.trapz for numerical integration)
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
