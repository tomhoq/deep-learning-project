from typing import Literal
import torch
from tqdm import tqdm
from models.yolo.utils.non_max_suppression import non_max_suppression
from sys import stdout



def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    S,
    C,
    box_format: Literal['midpoint', 'corners'] = 'midpoint',
    device="cuda",
    loss_function = None
):
    """
    Extracts predicted and true bounding boxes from a dataset loader using a given model.
    
    1. Sets the model to evaluation mode.
    2. Loops through the DataLoader to process batches of images and labels.
    3. Converts predictions and ground truth to bounding box format.
    4. Applies Non-Max Suppression (NMS) on predictions to filter redundant boxes.
    5. Collects and stores predicted and ground truth boxes for each image.
    6. Resets the model to training mode after processing.

    Returns:
        (tuple)
        - all_pred_boxes: List of predicted boxes with confidence scores.
        - all_true_boxes: List of ground truth boxes above a confidence threshold.
    """

    all_pred_boxes = []
    all_true_boxes = []
    valid_loss = 0

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    tq = tqdm(total=len(loader) * loader.batch_size, desc='Validation', file=stdout)

    for _, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        if loss_function:
            loss = loss_function(predictions, labels)
            valid_loss += loss * loader.batch_size

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels, S, C)
        bboxes = cellboxes_to_boxes(predictions, S, C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
        
        tq.update(loader.batch_size)

    tq.close()

    model.train()

    if loss_function:
        valid_loss /= len(loader.dataset)
        return all_pred_boxes, all_true_boxes, valid_loss
    else:
        return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S, C):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, C + 10)
    bboxes1 = predictions[..., C + 1:C + 5]
    bboxes2 = predictions[..., C + 6:C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S, C):
    """
    Converts raw output from the model into bounding boxes.

    Parameters:
        out (Tensor): Model's raw output (predictions).
        S (int): The grid size (default is 7x7 for a typical YOLO model).
    
    Returns:
        all_bboxes (List): A list of lists containing bounding box information for each example.
    """
    converted_pred = convert_cellboxes(out, S, C).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])