import time
import torch
import numpy as np
from .validation_metrics import Metrics
import logging


@torch.no_grad()
def validation(model: torch.nn.Module, loss_function, valid_loader, device, scheduler):
    times = []
    losses = []
    model.eval()

    metrics = Metrics(batch_size = valid_loader.batch_size) 
    
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        start_time = time.time()
        outputs = model(inputs)
        times.append(time.time() - start_time)

        loss = loss_function(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu())
    
    valid_loss = np.mean(losses)  # float
    valid_dice, valid_jaccard = metrics.get() # float
    inference_time_ms = (sum(times) / len(times)) * 1000

    if scheduler is not None:
        scheduler.step(valid_loss)

    print('    Valid loss: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}, , Inference time: {:.2f}ms'.format(valid_loss, valid_jaccard, valid_dice, inference_time_ms))

    return { 
        'valid_loss': valid_loss, 
        'jaccard': valid_jaccard.item(), 
        'dice': valid_dice.item(),
        'inference_time_ms': inference_time_ms,
    }