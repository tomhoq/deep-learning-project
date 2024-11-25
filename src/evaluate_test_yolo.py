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
from utils.helpers import DATA_DIR, draw_bboxes_on_image


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
  return I.astype(np.uint8)


_, axes = plt.subplots(batch_size, 1, figsize=(5, 5 * batch_size))
axes[0].set_title("Test output", fontsize=24, pad=20)

for idx in range(batch_size):
    pred_boxes = non_max_suppression(
        pred_bboxes[idx],
        iou_threshold=0.5, 
        threshold=0.4,
        box_format = 'midpoint',
    )

    image = images[idx].permute(1, 2, 0).numpy()
    image = normalize8(image)
    image = np.ascontiguousarray(image)

    draw_bboxes_on_image(image, pred_boxes)

    # Plotting the model output in the left column
    axes[idx].imshow(image)
    axes[idx].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(path.join(out_path, f"TEST_model.png"))
#####################################

print(f"[+] Completed")
