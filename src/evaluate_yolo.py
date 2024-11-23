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


########## Arguments ##########
# Check arguments
if len(argv) != 3:
    raise ValueError("Expected two or three arguments. Usage: python evaluate.py <out_path> <num_of_outputs>")

out_path = argv[1]
num_of_outputs = int(argv[2])
model_name = 'yolo'

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
_, val_dataset = get_yolo_train_val_datasets()

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
loader_iter = iter(val_loader)

S = 7
C = 1

# Display some images from loader
for i in range(num_of_outputs):
    plt.figure(figsize = (15,15))

    images, gt = next(loader_iter)
    gt = gt.data.cpu()
    images = images.data.cpu()
    out = model(images).data.cpu()

    batch_size = images.shape[0]
    true_bboxes = cellboxes_to_boxes(gt, S, C)
    bboxes = cellboxes_to_boxes(out, S, C)

    for idx in range(batch_size):
        boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=0.5, 
            threshold=0.4,
            box_format = 'midpoint',
        )

        image = images[idx]

        for items in boxes:
            Xmin  = int((items[0]-items[3]/2)*768)
            Ymin  = int((items[1]-items[2]/2)*768)
            Xmax  = int((items[0]+items[3]/2)*768)
            Ymax  = int((items[1]+items[2]/2)*768)
            cv2.rectangle(image,
                          (Xmin,Ymin),
                          (Xmax,Ymax),
                          (255,0,0),
                          thickness = 2)

        plt.subplot(4,4,i+1)
        plt.imshow(image)
        plt.title("No of ships = {}".format(i))

    plt.savefig(path.join(out_path, 'evaluation', f"model_vs_ground_truth_{i + 1}.png"))
#####################################
