import json
import numpy as np
import pandas as pd
from sys import argv
from os import path
import matplotlib.pyplot as plt
import torch
from models.yolo.dataset import get_yolo_train_val_datasets
from models.yolo.utils.helpers import cellboxes_to_boxes
from models.yolo.utils.non_max_suppression import non_max_suppression
from utils.get_model import get_model
import cv2

from utils.helpers import draw_bboxes_on_image


########## Arguments ##########
# Check arguments
if len(argv) != 3:
    raise ValueError("Expected two arguments. Usage: python evaluate.py <out_path> <num_of_outputs>")

out_path = argv[1]
num_of_outputs = int(argv[2])
model_name = 'yolo'

model = get_model(model_name)

print(f"\n[*] Evaluating {model_name} (running on {'GPU' if torch.cuda.is_available() else 'CPU'})")


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

# Load model
state = torch.load(str(model_path), map_location=device, weights_only=False)
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
model = model.to(device)
model.eval()


# Load data
_, val_dataset = get_yolo_train_val_datasets()

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
loader_iter = iter(val_loader)

S = 7
C = 1

##### Calculate metrics #####
from models.yolo.validation import validation as yolo_validation 
from models.yolo.loss import YoloLoss
loader_args = dict(shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, **loader_args)
comb_loss_metrics = yolo_validation(model, YoloLoss(), valid_loader, device, None)
# Write to a file
with open(path.join(out_path, 'evaluation', "metrics.txt"), "w") as file:
    file.write(json.dumps(comb_loss_metrics, sort_keys=True))
print("[+] Metrics saved")


##### Display some images from loader #####
for i in range(num_of_outputs):
    plt.figure(figsize = (15,15))

    images, gt = next(loader_iter)
    out = model(images.to(device))

    out = out.data.cpu()
    gt = gt.data.cpu()
    images = images.data.cpu()

    batch_size = images.shape[0]
    true_bboxes = cellboxes_to_boxes(gt, S, C)
    pred_bboxes = cellboxes_to_boxes(out, S, C)


    _, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    axes[0, 0].set_title("Model output", fontsize=24, pad=20)
    axes[0, 1].set_title("Ground truth", fontsize=24, pad=20)

    for idx in range(batch_size):
        pred_boxes = non_max_suppression(
            pred_bboxes[idx],
            iou_threshold=0.5, 
            threshold=0.4,
            box_format = 'midpoint',
        )
        true_boxes = true_bboxes[idx]

        image = images[idx]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8) 
        image = np.ascontiguousarray(image)

        pred_image = image.copy()
        true_image = image.copy()

        draw_bboxes_on_image(pred_image, pred_boxes)
        draw_bboxes_on_image(true_image, true_boxes)

        # Plotting the model output in the left column
        axes[idx, 0].imshow(pred_image)
        axes[idx, 0].axis('off')
        
        # Plotting the ground truths in the right column
        axes[idx, 1].imshow(true_image)
        axes[idx, 1].axis('off')


    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(path.join(out_path, 'evaluation', f"model_vs_ground_truth_{i + 1}.png"))
    print(f"[+] Image {i + 1} saved")
#####################################

print(f"[+] Evaluation completed")
