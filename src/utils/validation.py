import torch
import numpy as np
from .metrics import Metrics


@torch.no_grad()
def validation(model: torch.nn.Module, loss_function, valid_loader, device):
    losses = []
    model.eval()

    metrics = Metrics(batch_size = valid_loader.batch_size) 
    
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu())
    
    valid_loss = np.mean(losses)  # float
    valid_dice, valid_jaccard = metrics.get() # float

    print('    Valid loss: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}'.format(valid_loss, valid_jaccard, valid_dice))

    return { 
        'valid_loss': valid_loss, 
        'jaccard': valid_jaccard.item(), 
        'dice': valid_dice.item() 
    }