import torch
import torchvision


def get_resnet34_unet():
    resnet = torchvision.models.resnet34(weights='DEFAULT')
    
    # Substitute the FC output layer
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 1)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet