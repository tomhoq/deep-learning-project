from utils.losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss, DiceLoss
from utils.train import train
from models.unet.src.unet import UNet
from utils.dataset import AirbusDataset
import torch
from sys import argv
from utils.validation import validation as unet_validation

# Check arguments
if len(argv) != 4:
    raise ValueError("Expected exactly three arguments. Usage: python train.py <model> <loss_function> <out_path>.\n<model> = 'unet' | 'yolo'\n<loss_function> = 'bce' | 'jaccard' | 'dice'")



BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 4
LR = 1e-4
N_EPOCHS = 3

train_dataset = AirbusDataset(mode='train')
val_dataset = AirbusDataset(mode='validation')


####################
model_argv = argv[1]
model = None

print(f"\n[+] MODEL = {model_argv}")

if model_argv == 'unet':
    model = UNet()

elif model_argv == 'unet_resnet34':
    import segmentation_models_pytorch as smp
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    # if FREEZE_RESNET == True:
    #     for name, p in model.named_parameters():
    #         if "encoder" in name:
    #             p.requires_grad = False
####################


####################
loss = argv[2]
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
####################


train(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    loss_function = loss_function,
    validation_function = unet_validation,
    lr = LR,
    optimizer = torch.optim.Adam(model.parameters(), lr=LR),
    n_epochs = N_EPOCHS,
    train_batch_size = BATCH_SIZE_TRAIN,
    valid_batch_size = BATCH_SIZE_VALID,
    out_path = argv[3],
)