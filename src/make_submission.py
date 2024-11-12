import os
from sys import argv, stdout
import numpy as np
import pandas as pd
from skimage.morphology import binary_opening, disk
import torch
from tqdm import tqdm
from models.unet.src.unet import UNet
from utils.dataset import AirbusDataset
from utils.helpers import PATHS, multi_rle_encode
import torch.nn.functional as F


########## Arguments ##########
# Check arguments
if len(argv) != 3:
    raise ValueError("Expected exactly two arguments. Usage: python evaluate.py <model> <out_path>.\n<model> = 'unet' | 'yolo'")

out_path = argv[2]
# TODO model = UNet() if argv[1] == 'unet' else YOLO()
model = UNet()

print(f"\n[*] Making submission {'U-Net' if argv[1] == 'unet' else 'YOLO'} model")


########## Load model ##########
RUN_ID = 1
model_path = os.path.join(out_path, 'model.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[!] RUNNING ON {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

# Load model
state = torch.load(str(model_path), map_location=device, weights_only=False)
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
model = model.to(device)
model.eval()


########## Make submission ##########
list_img_test = os.listdir(PATHS['test'])
print('[*]', len(list_img_test), 'test images found\n')

# Create dataframe
test_df = pd.DataFrame({ 'ImageId': list_img_test, 'EncodedPixels': None })
ds = AirbusDataset(test_df, transform=None, mode='test')
loader = torch.utils.data.DataLoader(dataset=ds, shuffle=False, batch_size=2, num_workers=0)
    
out_pred_rows = []
for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Test', file=stdout)):
    inputs = inputs.to(device)
    outputs = model(inputs)

    for i, image_name in enumerate(paths):
        mask = F.sigmoid(outputs[i,0]).data.detach().cpu().numpy()
        cur_seg = binary_opening(mask>0.5, disk(2))
        cur_rles = multi_rle_encode(cur_seg)
        if len(cur_rles)>0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]
        
submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
res_path = os.path.join(out_path, 'submission.csv')
submission_df.to_csv(res_path, index=False)

print("\n[+] Done. Sample of submission.csv:")
print(submission_df.sample(10))