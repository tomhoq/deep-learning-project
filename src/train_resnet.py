from utils.losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss, DiceLoss
from utils.train import train
from models.unet.src.unet import UNet
import utils
import torch
from sys import argv
from models.resnet34unet.resnet34unet import get_resnet34_unet
from models.resnet34unet.dataset import AirbusDataset as ResnetDataset
from models.resnet34unet.validation import validation as resnet_validation
from utils.validation import validation as unet_validation

# Check arguments
if len(argv) != 2:
    raise ValueError("Expected exactly 1 arguments. Usage: python train.py <out_path>.")

out_path = argv[1]


print(f"\n[+] MODEL = resnet34")

model = get_resnet34_unet()
train_dataset = ResnetDataset(mode='train')
val_dataset = ResnetDataset(mode='validation')

loss_function = torch.nn.CrossEntropyLoss()
print('[+] Using CrossEntropyLoss')


BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32
N_EPOCHS = 5
WEIGHT_DECAY = 5e-4


torch.compile(model)


##### FINE TUNE HEAD #####
LR = 0.01

train(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = resnet_validation,
    lr = LR,
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR),
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    out_path = out_path,
)



##### FINE TUNE BACKBONE #####

for name, param in model.named_parameters():
    param.requires_grad = True

LR = 0.0001

train(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = resnet_validation,
    lr = LR,
    optimizer = torch.optim.Adam(params = model.parameters(), lr = LR),
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    out_path = out_path,
)