# From https://www.kaggle.com/code/alexj21/pytorch-eda-unet-from-scratch-finetuning/notebook
from torch import nn
import torch
import torch.nn.functional as F


class BCEJaccardWithLogitsLoss(nn.Module):
    def __init__(self, jaccard_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))
            
        loss = self.bce(outputs, targets)

        if self.jaccard_weight:
            targets = (targets == 1.0).float()
            targets = targets.view(-1)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)

            intersection = (targets * outputs).sum()
            union = outputs.sum() + targets.sum() 

            loss -= self.jaccard_weight * torch.log((intersection + self.smooth) / (union - intersection + self.smooth))

        return loss


class BCEDiceWithLogitsLoss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.smooth = smooth
        
    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))
            
        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = F.sigmoid(outputs)
        outputs = outputs.view(-1)

        intersection = (outputs * targets).sum()
        dice = 2.0 * (intersection + self.smooth)  / (targets.sum() + outputs.sum() + self.smooth)
        
        loss -= self.dice_weight * torch.log(dice)

        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply softmax if necessary
        y_pred = F.softmax(y_pred, dim=1) if y_pred.size(1) > 1 else torch.sigmoid(y_pred)
        
        # Flatten tensors for intersection computation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate Dice coefficient
        intersection = (y_pred * y_true).sum()
        dice_coeff = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        target = target.squeeze(1)  # Remove the channel dimension

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.0

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(self.dice_loss(input, target))
        return loss.mean()