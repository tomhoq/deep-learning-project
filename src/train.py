from utils.dataset import AirbusDataset, get_dataframes, get_transforms
from utils.helpers import PATHS
from utils.losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss
from utils.train_validation import train
from models.unet.src.unet import UNet
from torch.optim import Adam
import torch
from torch.nn import BCEWithLogitsLoss
from sys import argv

# Check arguments
if len(argv) != 4:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <model> <loss_function> <out_path>.\n<model> = 'unet' | 'yolo'\n<loss_function> = 'bce' | 'jaccard' | 'dice'")

# Train run number
RUN_ID = 1

# Parameters
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 4
LR = 1e-4
N_EPOCHS = 3

# Transforms
train_transform, val_transform = get_transforms()

# Initialize dataset
train_df, valid_df = get_dataframes()
train_dataset = AirbusDataset(train_df, transform=train_transform, mode='train')
val_dataset = AirbusDataset(valid_df, transform=val_transform, mode='validation')

print('\n[*] Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))
print(f"[*] Training {'U-Net' if argv[1] == 'unet' else 'YOLO'} model")

# Get loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_VALID, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())


# model = UNet() if argv[1] == 'unet' else YOLO()
model = UNet(input_channels = 3, output_classes = 1)

optimizer = Adam(model.parameters(), lr=LR)

loss_function = None
loss = argv[2]
if loss == 'bce':
    loss_function = BCEWithLogitsLoss()
    print('[+] Using BCE loss')
elif loss == 'jaccard':
    loss_function = BCEJaccardWithLogitsLoss()
    print('[+] Using Jaccard loss')
elif loss == 'jaccard2':
    loss_function = BCEJaccardWithLogitsLoss(jaccard_weight=5, smooth=1e-15)
    print('[+] Using Jaccard loss (jaccard_weight=5, smooth=1e-15)')
elif loss == 'dice':
    loss_function = BCEDiceWithLogitsLoss()
    print('[+] Using DICE loss')

train(
    model = model,
    train_loader = train_loader,
    valid_loader = val_loader,
    loss_function = loss_function,
    lr = LR,
    optimizer = optimizer,
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    fold = RUN_ID,
    out_path = argv[3],
)