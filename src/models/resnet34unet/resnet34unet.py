import torch
import torchvision
from torch import nn


def get_resnet34_unet():
    resnet = torchvision.models.resnet34(weights='DEFAULT')

    # Freeze all param to do transfer learning
    for name, param in resnet.named_parameters():
        param.requires_grad = False

    num_classes = 2  # [ship, no_ship]
    
    # Substitute the FC output layer
    resnet.fc = nn.Sequential(
                    nn.Linear(resnet.fc.in_features,128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

    # torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet