import torch
import torchvision
from torch import nn


def get_resnet34():
    resnet = torchvision.models.resnet34(weights='DEFAULT')

    num_classes = 1  # [ship / no_ship]
    resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(resnet.fc.in_features, 1),
        nn.Sigmoid()
    )

    return resnet