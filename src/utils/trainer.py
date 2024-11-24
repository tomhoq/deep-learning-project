import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
from os import path
from sys import stdout
# import wandb
import pandas as pd
import matplotlib.pyplot as plt


class Trainer():
    def __init__(
        self,
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
        num_workers_per_gpu=2,
        scheduler=None,
        start_train_immediately = True
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f'''Starting training:
            Epochs:           {n_epochs}
            Train Batch size: {train_batch_size}
            Valid Batch size: {valid_batch_size}
            Learning rate:    {lr}
            Training size:    {len(train_dataset)}
            Validation size:  {len(val_dataset)}
            Optimizer:        {type (optimizer).__name__}
            Scheduler:        {'Yes' if scheduler is not None else 'No'}
            Device:           {device.type}
        ''')

        ##### LOADERS #####
        num_gpus = torch.cuda.device_count()
        loader_args = dict(shuffle=True, num_workers=num_workers_per_gpu*num_gpus, pin_memory=torch.cuda.is_available())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, **loader_args)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=valid_batch_size, **loader_args)

        ##### RESTORE #####
        self.model_path = Path(path.join(out_path, 'model.pt'))
        if self.model_path.exists():
            state = torch.load(str(self.model_path), map_location=device, weights_only=False)
            self.epoch = state['epoch']
            self.step = state['step']
            model.load_state_dict(state['model'])
            print('[*] Restored model, epoch {}, step {:,}'.format(self.epoch, self.step))
        else:
            self.epoch = 1
            self.step = 0

        ##### LOGGING #####
        self.REPORT_EACH = 50
        self.log = open(path.join(out_path, 'train.log'), 'at', encoding='utf8')
        # self.experiment = wandb.init(project='regi18-train', resume='allow', anonymous='must')
        # self.experiment.config.update(
        #     dict(epochs=n_epochs, train_batch_size=train_batch_size, valid_batch_size=valid_batch_size, learning_rate=lr)
        # )

        ##### VARIABLES #####
        self.model = model
        self.loss_function = loss_function
        self.validation_function = validation_function
        self.lr = lr
        self.optimizer = optimizer
        self.out_path = out_path
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.n_epochs = n_epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

        if start_train_immediately:
            self.train()


    def train(self):
        for epoch in range(self.epoch, self.n_epochs + 1):
            self.epoch = epoch
            self.model.train()
            losses = []

            # Progress bar
            tq = tqdm(
                total=len(self.train_loader) * self.train_batch_size, 
                desc=f"Epoch {epoch}/{self.n_epochs}", 
                file=stdout, 
                unit='img'
            )

            try:
                mean_loss = 0

                for i, (inputs, targets) in enumerate(self.train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    ### FORWARD AND BACK PROP ###
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())

                    ### UPDATE MODEL PARAMETERS ###
                    self.optimizer.step()

                    ### LOGGING ###
                    self._log_training(i, tq, losses)

                # Logging (conclude logging for this epoch)
                self._log_event(loss=mean_loss)
                tq.close()

                ### CHECKPOINT ###
                self._save(epoch + 1)

                ### VALIDATION AND LOGGING ###
                comb_loss_metrics = self.validation_function(self.model, self.loss_function, self.valid_loader, self.device, self.scheduler)
                self._log_event(**comb_loss_metrics)
                # self._update_wandb(comb_loss_metrics)
                self._plot_progress()

            # Save model if training is interrupted
            except KeyboardInterrupt:
                tq.close()
                print('Ctrl+C, saving snapshot')
                self._save(epoch)
                print('done.')
                return

        print("\n[+] Finished training")


    #############################################################


    # def _update_wandb(self, comb_loss_metrics):
    #     histograms = {}
    #     for tag, value in self.model.named_parameters():
    #         tag = tag.replace('/', '.')
    #         if not (torch.isinf(value) | torch.isnan(value)).any():
    #             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
    #         if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
    #             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

    #     self.experiment.log({
    #         'learning rate': self.optimizer.param_groups[0]['lr'],
    #         **comb_loss_metrics,
    #         'step': self.step,
    #         'epoch': self.epoch,
    #         **histograms
    #     })


    def _plot_progress(self):
        log_file = path.join(self.out_path, 'train.log')
        logs = pd.read_json(log_file, lines=True)

        # Steps vs training loss plot
        fig, ax1 = plt.subplots()

        ax1.plot(
            logs.step[logs.loss.notnull()],
            logs.loss[logs.loss.notnull()],
            label="on training set",
            color='tab:blue'
        )
        
        # Steps vs validation loss plot
        ax1.plot(
            logs.step[logs.valid_loss.notnull()],
            logs.valid_loss[logs.valid_loss.notnull()],
            label="on validation set",
            color='tab:orange'
        )
        
        ax1.set_xlabel('step')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='center left')

        # Add secondary x-axis for 'epoch'
        ax2 = ax1.twiny()

        # Use the 'epoch' values directly from the logs DataFrame
        ax2.set_xlim(ax1.get_xlim())  # Make sure the limits of the secondary x-axis match the first x-axis
        ax2.set_xticks(logs.step[logs.epoch.notnull()])  # Set ticks based on the available epoch values
        ax2.set_xticklabels(logs.epoch[logs.epoch.notnull()])  # Display the epoch values as tick labels
        ax2.set_xlabel('Epoch')

        # Tight layout and saving the figure
        plt.tight_layout()
        plt.savefig(path.join(self.out_path, 'loss.png'))


    def _log_training(self, i, tq, losses):
        self.step += 1
        tq.update(self.train_batch_size)

        mean_loss = np.mean(losses[-self.REPORT_EACH:])
        tq.set_postfix(loss='{:.5f}'.format(mean_loss))

        if i and i % self.REPORT_EACH == 0:
            self._log_event(loss=mean_loss)


    def _save(self, ep):
        return torch.save({
            'model': self.model.state_dict(),
            'epoch': ep,
            'step': self.step,
        }, str(self.model_path))


    def _log_event(self, **data):
        """
        Helper to log data into file
        """
        data['step'] = self.step
        data['epoch'] = self.epoch
        data['dt'] = datetime.now().isoformat()
        self.log.write(json.dumps(data, sort_keys=True))
        self.log.write('\n')
        self.log.flush()
