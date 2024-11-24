from models.yolo.dataset import get_yolo_train_val_datasets
from models.yolo.loss import YoloLoss
from models.yolo.utils.dice import match_and_calculate_dice
from models.yolo.utils.helpers import get_bboxes
from utils.get_model import get_model
from utils.trainer import Trainer
import torch
from sys import argv
from models.yolo.validation import validation as yolo_validation 
import logging
from torch import nn


# Check arguments
if len(argv) != 2:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <out_path>")

out_path = argv[1]

print(f"\nMODEL = YOLOv1")


##### LOSS #####
loss_function = YoloLoss()
print('Using YoloLoss')




train_dataset, val_dataset = get_yolo_train_val_datasets()



##### TRAIN #####

BATCH_SIZE = 32
LR = 2e-5
N_EPOCHS = 20

model = get_model("yolo")

####
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)
####

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5, mode='max')
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

Trainer(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = yolo_validation,
    lr = LR,
    optimizer = optimizer,
    scheduler = scheduler,
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE,
    valid_batch_size = BATCH_SIZE,
    out_path = out_path,
)



##### Calculate DICE #####

valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2*torch.cuda.device_count(), pin_memory=torch.cuda.is_available())
pred_boxes, target_boxes, valid_loss = get_bboxes(
    valid_loader, 
    model, 
    iou_threshold=0.5,
    threshold=0.4,
    S = 7,
    C = 1,
    device = "cuda", 
    loss_function = loss_function
)
# Take only [x1,y1,x2,y2] from [train_idx, class_pred, prob_score, x1, y1, x2, y2]
pred_bboxes = torch.Tensor(pred_boxes)[..., 3:7]
target_bboxes = torch.Tensor(target_boxes)[..., 3:7]

dice_score = match_and_calculate_dice(pred_bboxes, target_bboxes)

print("DICE score: {:.5f}".format(dice_score))