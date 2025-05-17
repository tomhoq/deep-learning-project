import argparse
from models.yolo.dataset import get_yolo_train_val_datasets
from models.yolo.loss import YoloLoss
from utils.get_model import get_model
from utils.trainer import Trainer
import torch
from sys import argv
from models.yolo.validation import validation as yolo_validation 



##### ARGS #####
parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

parser.add_argument('--out-path', '-o', metavar="O", dest='out_path', type=str, help='Job out path')
parser.add_argument('--epochs', '-e', metavar='E', type=int, help='Number of epochs')
parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, help='Learning rate', dest='lr')
parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, help='Weight decay', dest='weight_decay')
parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, help='Batch size')
parser.add_argument('--lambda-noobj', dest='lambda_noobj', metavar='LN', type=float, help='lambda_noobj')
parser.add_argument('--lambda-coord', dest='lambda_coord', metavar='LC', type=float, help='lambda_coord')

args = parser.parse_args()



##### DATA #####
print(f"\nMODEL = YOLOv1")
train_dataset, val_dataset = get_yolo_train_val_datasets()


##### LOSS #####
loss_function = YoloLoss(lambda_noobj=args.lambda_noobj, lambda_coord=args.lambda_coord)
print(f'Using YoloLoss(lambda_noobj={args.lambda_noobj}, lambda_coord={args.lambda_coord})')



##### TRAIN #####

BATCH_SIZE = args.batch_size
LR = args.lr
N_EPOCHS = args.epochs
WEIGHT_DECAY = args.weight_decay

print(f"WEIGHT_DECAY = {args.weight_decay}")

model = get_model("yolo")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, mode='max')

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
    out_path = args.out_path,
)
