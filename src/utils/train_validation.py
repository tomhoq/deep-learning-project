import torch
import random
from torch import nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
from .metrics import metrics
from os import path
from sys import stdout


def write_event(log, step: int, **data):
    """
    Helper to log data into file
    """
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(model: nn.Module, train_loader, valid_loader, loss_function, lr, optimizer, out_path, train_batch_size=16, valid_batch_size=4, n_epochs=1, fold=1):
    """
    From https://github.com/ternaus/robot-surgery-segmentation

    Trains the given nn model. It can also stop and restart training from a file.
    Everything is logged into files.
    """
    
    # Restore model or start from scratch
    model_path = Path(path.join(out_path, 'model_{fold}.pt'.format(fold=fold)))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    # Create save function
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    # Init logging
    report_each = 50
    log = open(path.join(out_path, 'train_{fold}.log'.format(fold=fold)), 'at', encoding='utf8')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[!] TRAINING USING {'GPU' if torch.cuda.is_available() else 'CPU'}")
    model = model.to(device)

    print(f"[*] Training options: train_batch_size={train_batch_size}, valid_batch_size={valid_batch_size}, n_epochs={n_epochs}\n\n")

    for epoch in range(epoch, n_epochs + 1):

        model.train()

        # For logging
        random.seed()
        tq = tqdm(total=len(train_loader) *  train_batch_size, file=stdout)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []

        try:
            mean_loss = 0

            for i, (inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(device), targets.to(device)

                ### FORWARD AND BACK PROP ###
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                optimizer.zero_grad()
                loss.backward()

                ### UPDATE MODEL PARAMETERS ###
                optimizer.step()
                
                ### LOGGING ###
                step += 1
                batch_size = inputs.size(0)
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            # Logging (conclude logging for this epoch)
            write_event(log, step, loss=mean_loss)
            tq.close()

            # Save model
            save(epoch + 1)
            
            # Validation and logging
            comb_loss_metrics = validation(model, loss_function, valid_loader, metrics(batch_size = valid_batch_size), device)
            write_event(log, step, **comb_loss_metrics)

        # Save model if training is interrupted
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

    print("\n[+] Finished training")
        

def validation(model: nn.Module, loss_function, valid_loader, metrics, device):
    losses = []
    model.eval()
    
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu())
    
    valid_loss = np.mean(losses)  # float
    valid_dice, valid_jaccard = metrics.get() # float

    print('Valid loss: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}'.format(valid_loss, valid_jaccard, valid_dice))

    return { 
        'valid_loss': valid_loss, 
        'jaccard': valid_jaccard.item(), 
        'dice': valid_dice.item() 
    }
