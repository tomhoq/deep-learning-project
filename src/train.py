from utils.data_augmentation import CenterCrop, DualCompose, HorizontalFlip, RandomCrop, VerticalFlip
from utils.dataset import AirbusDataset, get_dataframes
from utils.helpers import PATHS
from utils.train_validation import train
from models.unet.src.unet import UNet
from torch.optim import Adam
import torch
from torch.nn import BCEWithLogitsLoss
from sys import argv

# Expects the name of the model (unet or yolo) as argument
if len(argv) != 2:  # Expect exactly one argument (plus the script name)
    raise ValueError("Expected exactly one argument. Usage: python test.py <model>.\n<model> = 'unet' | 'yolo'")

# Train run number
RUN_ID = 1

# Parameters
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 4
LR = 1e-4
N_EPOCHS = 3

# Transforms
train_transform = DualCompose([HorizontalFlip(), VerticalFlip(), RandomCrop((256,256,3))])
val_transform = DualCompose([CenterCrop((512,512,3))])

# Initialize dataset
train_df, valid_df = get_dataframes()
train_dataset = AirbusDataset(train_df, transform=train_transform, mode='train')
val_dataset = AirbusDataset(valid_df, transform=val_transform, mode='validation')

print('[*] Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))

# Get loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_VALID, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())


# model = UNet() if argv[1] == 'unet' else YOLO()
model = UNet(input_channels = 3, output_classes = 1)

optimizer = Adam(model.parameters(), lr=LR)
loss_function = BCEWithLogitsLoss()

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
    fold = RUN_ID
)