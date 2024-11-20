from utils.train import train
from models.unet.src.unet import UNet
import torch
from sys import argv
from models.resnet34.resnet34 import get_resnet34
from models.resnet34.dataset import AirbusDataset as ResnetDataset
from models.resnet34.validation import validation as resnet_validation

# Check arguments
if len(argv) != 2:
    raise ValueError("Expected exactly 1 arguments. Usage: python train.py <out_path>.")

out_path = argv[1]


print(f"\n[+] MODEL = resnet34")

model = get_resnet34()
train_dataset = ResnetDataset(mode='train')
val_dataset = ResnetDataset(mode='validation')

loss_function = torch.nn.BCELoss()
print('[+] Using BCELoss')


BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32
N_EPOCHS = 5
WEIGHT_DECAY = 5e-4


torch.compile(model)


LR = 2e-3

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
