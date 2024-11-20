import pandas as pd
from sys import argv
from os import path
import matplotlib.pyplot as plt
import torch
from models.unet.src.unet import UNet
from utils.dataset import AirbusDataset, get_dataframes, get_transforms
from utils.get_model import get_model
from utils.helpers import compare_model_outputs_with_ground_truths


########## Arguments ##########
# Check arguments
if len(argv) != 3 and len(argv) != 4:
    raise ValueError("Expected two or three arguments. Usage: python evaluate.py <model> <out_path> <num_of_outputs = 1>.\n<model> = 'unet' | 'yolo'")

model_name = argv[1]
out_path = argv[2]

num_of_outputs = 1
if len(argv) == 4:
    num_of_outputs = int(argv[3])

model = get_model(model_name)

print(f"\n[*] Evaluating {model_name} model")


########## Plot losses ##########
RUN_ID = 1
log_file = path.join(out_path, 'train.log')
logs = pd.read_json(log_file, lines=True)

# plt.figure(figsize=(20,6))
# Steps vs training loss plot
plt.plot(
    logs.step[logs.loss.notnull()],
    logs.loss[logs.loss.notnull()],
    label="on training set"
)
# Steps vs validation loss plot
plt.plot(
    logs.step[logs.valid_loss.notnull()],
    logs.valid_loss[logs.valid_loss.notnull()],
    label = "on validation set"
)
plt.xlabel('step')
plt.legend(loc='center left')
plt.tight_layout()
plt.savefig(path.join(out_path, 'evaluation', 'loss.png'))
#####################################


########## Model inference ##########
model_path = path.join(out_path, 'model.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[!] RUNNING ON {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

# Load model
state = torch.load(str(model_path), map_location=device, weights_only=False)
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
model = model.to(device)
model.eval()

# Load data
train_df, valid_df = get_dataframes()
train_transform, val_transform = get_transforms()
val_dataset = AirbusDataset(valid_df, transform=val_transform, mode='validation')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
loader_iter = iter(val_loader)

# Display some images from loader
for i in range(num_of_outputs):
    images, gt = next(loader_iter)
    gt = gt.data.cpu()
    images = images.to(device)
    out = model(images)
    out = ((out > 0).float()) * 255
    images = images.data.cpu()
    out = out.data.cpu()

    compare_model_outputs_with_ground_truths(images, gt, out)
    plt.savefig(path.join(out_path, 'evaluation', f"model_vs_ground_truth_{i + 1}.png"))
#####################################