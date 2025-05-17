import math
import os
import numpy as np
import pandas as pd
from sys import argv
from os import path
import matplotlib.pyplot as plt
import torch
from models.unet.src.utils.dataset import AirbusUnetDataset
from models.yolo.utils.helpers import cellboxes_to_boxes
from models.yolo.utils.non_max_suppression import non_max_suppression
from utils.data_augmentation import DualCompose, Resize
from utils.get_model import get_model
from utils.helpers import DATA_DIR, draw_bboxes_on_image, get_image_from_tensor_and_masks



########## Arguments ##########
# Check arguments
if len(argv) != 2:
    raise ValueError("Expected two arguments. Usage: python evaluate.py <out_path>")

out_path = argv[1]
model_name = 'yolo'

model = get_model(model_name)

print(f"[*] Evaluating {model_name} (running on {'GPU' if torch.cuda.is_available() else 'CPU'})")



########## Model ##########
model_path = path.join(out_path, 'model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(str(model_path), map_location=device, weights_only=False)
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
model = model.to(device)
model.eval()


########## Data ##########
my_path = f"{DATA_DIR}/my_test_subset"
list_img_test = os.listdir(my_path)
test_df = pd.DataFrame({ 'ImageId': list_img_test, 'EncodedPixels': None })
t = DualCompose([Resize((448,448))])
# Keep the unet dataset because we're just using as a facade, the 'test' mode deoesn't do much
ds = AirbusUnetDataset(test_df, mode='test', transform=t, path=my_path)
loader = torch.utils.data.DataLoader(dataset=ds, shuffle=False, batch_size=len(list_img_test), num_workers=0)
loader_iter = iter(loader)


########## Evaluate ##########
S = 7
C = 1

# Display some images from loader
plt.figure(figsize = (15,15))

images, gt = next(loader_iter)
out = model(images.to(device))

out = out.data.cpu()
images = images.data.cpu()

batch_size = images.shape[0]
pred_bboxes = cellboxes_to_boxes(out, S, C)



########## Plot ##########
def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  I = I.astype(np.uint8)
  return np.ascontiguousarray(I)


cols = 2
rows = math.ceil(batch_size / cols)  # Calculate the number of rows needed
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten() if rows > 1 else [axes]

# plt.suptitle("Test output", fontsize=24)

for idx in range(batch_size):
    pred_boxes = non_max_suppression(
        pred_bboxes[idx],
        iou_threshold=0.5, 
        threshold=0.4,
        box_format = 'midpoint',
    )

    image, _ = get_image_from_tensor_and_masks(images[idx], [])
    image = np.ascontiguousarray(image)

    # Keep only boxes with a certain confidence score
    CONFIDENCE_SCORE_THRESHOLD = 0.5
    pred_boxes = [box for box in pred_boxes if box[1] > CONFIDENCE_SCORE_THRESHOLD]

    draw_bboxes_on_image(image, pred_boxes)

    # Plotting the model output in the left column
    axes[idx].imshow(image)
    axes[idx].axis('off')

# Turn off unused subplots
for i in range(batch_size, len(axes)):
    axes[i].axis('off')

# plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.tight_layout()
plt.savefig(path.join(out_path, f"TEST_model.png"))
#####################################

print(f"[+] Completed")
