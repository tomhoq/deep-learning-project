from typing import Literal
from torch.nn import Module


def get_model(model_name: Literal['unet', 'unet_resnet34']) -> Module:

    if model_name == 'unet':
        from models.unet.src.unet import UNet
        return UNet()

    elif model_name  == 'unet_resnet34':
        import segmentation_models_pytorch as smp
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        # if FREEZE_RESNET == True:
        #     for name, p in model.named_parameters():
        #         if "encoder" in name:
        #             p.requires_grad = False
        return model
