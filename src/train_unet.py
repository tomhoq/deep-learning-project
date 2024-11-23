from models.yolo.loss import YoloLoss
from utils.get_model import get_model
from utils.losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss, DiceLoss, MixedLoss
from utils.train import train
from models.unet.src.unet import UNet
from utils.dataset import AirbusDataset, get_dataframes
import torch
from sys import argv
from utils.validation import validation as unet_validation

# Check arguments
if len(argv) != 4:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <model> <loss_function> <out_path>.")

model_argv = argv[1]
loss = argv[2]
out_path = argv[3]

print(f"\n[+] MODEL = {model_argv}")


df = get_dataframes()
train_dataset = AirbusDataset(mode='train', in_df=df['train'])
val_dataset = AirbusDataset(mode='validation', in_df=df['validation'])


####################
loss_function = None

if loss == 'bce':
    loss_function = torch.nn.BCEWithLogitsLoss()
    print('[+] Using BCE loss')
elif loss == 'cross_entropy':
    loss_function = torch.nn.CrossEntropyLoss()
    print('[+] Using CrossEntropyLoss')
#
elif loss == 'jaccard':
    loss_function = BCEJaccardWithLogitsLoss()
    print('[+] Using Jaccard loss')
#
elif loss == 'jaccard2':
    loss_function = BCEJaccardWithLogitsLoss(jaccard_weight=5, smooth=1e-15)
    print('[+] Using Jaccard loss (jaccard_weight=5, smooth=1e-15)')
#
elif loss == 'dice':
    loss_function = BCEDiceWithLogitsLoss()
    print('[+] Using DICE loss')
#
elif loss == 'dice_no_bce':
    loss_function = DiceLoss()
    print('[+] Using DICE loss (but without BCE)')
#
elif loss == 'mixed':
    loss_function = MixedLoss(10.0, 2.0)
    print('[+] Using MixedLoss(10.0, 2.0)')
####################


##### TRAIN #####

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 4
LR = 1e-4
N_EPOCHS = 3

model = get_model(model_argv)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max')

train(
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
    out_path = out_path,
)