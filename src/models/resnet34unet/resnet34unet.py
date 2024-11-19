import torch
import torchvision


def get_resnet34_unet():
    resnet = torchvision.models.resnet34(weights='DEFAULT')
    
    # Substitute the FC output layer
    num_classes = 2  # [ship, no_ship]
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet