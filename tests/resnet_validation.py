from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch
from models.resnet34.resnet34 import get_resnet34
from models.resnet34.dataset import AirbusDataset as ResnetDataset
from models.resnet34.validation import validation as resnet_validation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

model = get_resnet34()

val_dataset = ResnetDataset(mode='validation')
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2*num_gpus, pin_memory=torch.cuda.is_available())

loss_function = torch.nn.BCELoss()


# model_path = Path('/zhome/82/4/212615/deep-learning-project/job_out/resnet34/23232852/model.pt')
# if model_path.exists():
#     state = torch.load(str(model_path), map_location=device, weights_only=False)
#     epoch = state['epoch']
#     step = state['step']
#     model.load_state_dict(state['model'])
#     print('[*] Restored model, epoch {}, step {:,}'.format(epoch, step))


print("[+] Validating...")

comb_loss_metrics = resnet_validation(model.to(device), loss_function, valid_loader, device)
print(comb_loss_metrics)