from typing import Literal
import torch
from torch.nn import Module



def get_model(model_name: Literal['unet', 'unet_resnet34', 'unet34', 'yolo']) -> Module:
    #---------------------------------------- 
    if model_name == 'unet':
        from models.unet.src.unet import UNet
        model = UNet()
    # ----------------------------------------
    elif model_name  == 'yolo':
        from models.yolo.yolo import YOLO
        model = YOLO()
    # ----------------------------------------
    elif model_name  == 'unet34':
        from models.unet34.unet34 import Unet34
        model = Unet34()
        model.freeze_resnet()
    # ----------------------------------------
    elif model_name  == 'unet_resnet34':
        import segmentation_models_pytorch as smp
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    #---------------------------------------- 

    torch.compile(model)
    return model
