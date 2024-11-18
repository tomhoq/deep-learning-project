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
if len(argv) != 4:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <model> <loss_function> <out_path>.\n<model> = 'unet' | 'yolo'\n<loss_function> = 'bce' | 'jaccard' | 'dice'")


####################
model_argv = argv[1]
model = None

print(f"[+] MODEL = {model_argv}")

if model_argv == 'unet':
    model = UNet()
    train_dataset = utils.dataset.AirbusDataset(mode='train')
    val_dataset = utils.dataset.AirbusDataset(mode='validation')
    BATCH_SIZE_TRAIN = 16
    BATCH_SIZE_VALID = 4
    LR = 1e-4
    N_EPOCHS = 3
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    val_fun = unet_validation

elif model_argv == 'resnet34unet':
    model = get_resnet34_unet()
    train_dataset = ResnetDataset(mode='train')
    val_dataset = ResnetDataset(mode='validation')
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALID = 64
    LR = 1e-4
    N_EPOCHS = 10
    WEIGHT_DECAY = 5e-4
    # params_1x are the parameters of the network body, i.e., of all layers except the FC layers
    params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    optimizer = torch.optim.Adam([{ 'params':params_1x }, { 'params': model.fc.parameters(), 'lr': LR*10 }], lr=LR, weight_decay=WEIGHT_DECAY)
    val_fun = resnet_validation
####################


####################
loss = argv[2]
loss_function = None

if loss == 'bce':
    loss_function = torch.nn.BCEWithLogitsLoss()
    print('[+] Using BCE loss')
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
####################


train(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = val_fun,
    lr = LR,
    optimizer = optimizer,
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    out_path = argv[3],
)