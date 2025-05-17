import argparse
from models.unet.src.utils.dataset import get_unet_train_val_datasets
from models.yolo.loss import YoloLoss
from utils.get_model import get_model
from utils.losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss, DiceLoss, MixedLoss
from utils.trainer import Trainer
from models.unet.src.unet import UNet
import torch
from sys import argv
from models.unet.src.utils.validation import validation as unet_validation
import logging


model_argv = 'unet'
loss = 'bce'

print(f"\nMODEL = {model_argv}")

train_dataset, val_dataset = get_unet_train_val_datasets()


####################
loss_function = None

if loss == 'bce':
    loss_function = torch.nn.BCEWithLogitsLoss()
    print('Using BCE loss')
elif loss == 'cross_entropy':
    loss_function = torch.nn.CrossEntropyLoss()
    print('Using CrossEntropyLoss')
#
elif loss == 'jaccard':
    loss_function = BCEJaccardWithLogitsLoss()
    print('Using Jaccard loss')
#
elif loss == 'jaccard2':
    loss_function = BCEJaccardWithLogitsLoss(jaccard_weight=5, smooth=1e-15)
    print('Using Jaccard loss (jaccard_weight=5, smooth=1e-15)')
#
elif loss == 'dice':
    loss_function = BCEDiceWithLogitsLoss()
    print('Using DICE loss')
#
elif loss == 'dice_no_bce':
    loss_function = DiceLoss()
    print('Using DICE loss (but without BCE)')
#
elif loss == 'mixed':
    loss_function = MixedLoss(10.0, 2.0)
    print('Using MixedLoss(10.0, 2.0)')
####################



##### ARGS #####

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

parser.add_argument('--out-path', '-o', metavar="O", dest='out_path', type=str, help='Job out path')
parser.add_argument('--epochs', '-e', metavar='E', type=int, help='Number of epochs')
parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, help='Learning rate', dest='lr')
parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, help='Weight decay', dest='weight_decay')
parser.add_argument('--batch-size-train', '-bt', dest='batch_size_train', metavar='BT', type=int, help='Batch size train')
parser.add_argument('--batch-size-valid', '-bv', dest='batch_size_valid', metavar='BV', type=int, help='Batch size valid')

args = parser.parse_args()



##### TRAIN #####

BATCH_SIZE_TRAIN = args.batch_size_train
BATCH_SIZE_VALID = args.batch_size_valid
LR = args.lr
N_EPOCHS = args.epochs

model = get_model(model_argv)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=1, mode='max')

print(f"WEIGHT_DECAY = {args.weight_decay}")

Trainer(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = unet_validation,
    lr = LR,
    optimizer = torch.optim.Adam(model.parameters(), lr=LR),
    scheduler = scheduler,
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    out_path = args.out_path,
)