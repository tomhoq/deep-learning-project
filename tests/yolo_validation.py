import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.yolo.loss import YoloLoss
from models.yolo.dataset import get_yolo_train_val_datasets
from utils.get_model import get_model
import torch
from models.resnet34.dataset import AirbusDataset as ResnetDataset
from models.yolo.validation import validation as yolo_validation 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

_, val_dataset = get_yolo_train_val_datasets()
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2*num_gpus, pin_memory=torch.cuda.is_available())

print("[+] Validating...")

comb_loss_metrics = yolo_validation(model = get_model('yolo').to(device), loss_function = YoloLoss(), valid_loader = valid_loader, device = device, scheduler=None)
print(comb_loss_metrics)