import torch
import random
from torch import nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
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


def train(
        model: nn.Module, 
        train_dataset, 
        val_dataset, 
        loss_function, 
        validation_function, 
        lr, 
        optimizer, 
        out_path, 
        train_batch_size, 
        valid_batch_size, 
        n_epochs, 
        num_workers_per_gpu = 2,
        scheduler = None
    ):
    """
    Trains the given nn model. It can also stop and restart training from a file.
    Everything is logged into files.
    """

    print('[*] Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))

    # Get loaders
    num_gpus = torch.cuda.device_count()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers_per_gpu*num_gpus, pin_memory=torch.cuda.is_available())
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=num_workers_per_gpu*num_gpus, pin_memory=torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[!] TRAINING USING {'GPU' if torch.cuda.is_available() else 'CPU'}")

    torch.compile(model)

    # Restore model or start from scratch
    model_path = Path(path.join(out_path, 'model.pt'))
    if model_path.exists():
        state = torch.load(str(model_path), map_location=device, weights_only=False)
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('[*] Restored model, epoch {}, step {:,}'.format(epoch, step))
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
    log = open(path.join(out_path, 'train.log'), 'at', encoding='utf8')
    
    model = model.to(device)

    is_sched = 'yes' if scheduler is not None else 'no'
    print(f"[*] Training options: lr={lr}, scheduler={is_sched}, train_batch_size={train_batch_size}, valid_batch_size={valid_batch_size}, n_epochs={n_epochs}\n")

    for epoch in range(epoch, n_epochs + 1):

        model.train()

        # For logging
        random.seed()
        tq = tqdm(total=len(train_loader) *  train_batch_size, file=stdout)
        tq.set_description('Epoch {}'.format(epoch))
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
            comb_loss_metrics = validation_function(model, loss_function, valid_loader, device, scheduler)
            write_event(log, step, **comb_loss_metrics)

        # Save model if training is interrupted
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

    print("\n[+] Finished training")
