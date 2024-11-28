import math
import os
import pandas as pd
from sys import argv
from os import path
import matplotlib.pyplot as plt
import torch
from models.unet.src.utils.dataset import AirbusUnetDataset, get_unet_train_val_datasets
from utils.data_augmentation import CenterCrop, DualCompose, Resize
from utils.get_model import get_model
from utils.helpers import DATA_DIR, PATHS, compare_model_outputs_with_ground_truths, get_image_from_tensor_and_masks, mask_overlay


########## Arguments ##########
# Check arguments
if len(argv) != 3 and len(argv) != 3:
    raise ValueError("Expected two or three arguments. Usage: python evaluate.py <model> <out_path>")


model_name = argv[1]
out_path = argv[2]

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
# t = DualCompose([CenterCrop((448,448,3))])
t = DualCompose([Resize((448,448))])
# Keep the unet dataset because we're just using as a facade, the 'test' mode deoesn't do much
ds = AirbusUnetDataset(test_df, mode='test', transform=t, path=my_path)
loader = torch.utils.data.DataLoader(dataset=ds, shuffle=False, batch_size=len(list_img_test), num_workers=0)
batch_size = loader.batch_size
loader_iter = iter(loader)


########## Evaluate ##########
images, gt = next(loader_iter)
images = images.to(device)
out = model(images)
out = ((out > 0).float()) * 255
images = images.data.cpu()
out = out.data.cpu()


########## Plot ##########
cols = 2
rows = math.ceil(batch_size / cols)  # Calculate the number of rows needed
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten() if rows > 1 else [axes]

# plt.suptitle("Test output", fontsize=24)


for idx, (img, out) in enumerate(zip(images, out)):
    # Convert to image
    img, [out] = get_image_from_tensor_and_masks(img, [out])

    # Overlay image with masks
    out_overlay = mask_overlay(img, out)

    # Plotting the model output in the left column
    axes[idx].imshow(out_overlay)
    axes[idx].axis('off')

# Turn off unused subplots
for i in range(batch_size, len(axes)):
    axes[i].axis('off')

# plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.tight_layout()
plt.savefig(path.join(out_path, f"TEST_model.png"))



print("[+] Finished evaluating")