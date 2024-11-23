from models.yolo.dataset import get_yolo_train_val_datasets
from models.yolo.loss import YoloLoss
from utils.get_model import get_model
from utils.train import train
import torch
from sys import argv
from models.yolo.validation import validation as yolo_validation 


# Check arguments
if len(argv) != 2:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <out_path>")

out_path = argv[1]

print(f"\n[+] MODEL = YOLOv1")


##### LOSS #####
loss_function = YoloLoss()
print('[+] Using YoloLoss')




train_dataset, val_dataset = get_yolo_train_val_datasets()



##### TRAIN #####

BATCH_SIZE = 64
LR = 2e-5
N_EPOCHS = 20

model = get_model("yolo")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max')

train(
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