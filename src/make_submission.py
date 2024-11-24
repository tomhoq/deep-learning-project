import os
from sys import argv, stdout
from skimage.morphology import binary_opening, disk
import torch
import pandas as pd
from tqdm import tqdm
from models.unet.src.utils.dataset import AirbusUnetDataset
from utils.get_model import get_model
from utils.helpers import PATHS, multi_rle_encode
import torch.nn.functional as F
import logging


########## Arguments ##########
# Check arguments
if len(argv) != 3:
    raise ValueError("Expected exactly two arguments. Usage: python evaluate.py <model> <out_path>")

model_name = argv[1]
out_path = argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_img_test = os.listdir(PATHS['test'])

print(f'''Making submission:
    Model:   {model_name}
    Device:  {device.type}
    Samples: {len(list_img_test)}
''')


########## Load model ##########
model_path = os.path.join(out_path, 'model.pt')

print(f"RUNNING ON {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

# Load model
model = get_model(model_name)
state = torch.load(str(model_path), map_location=device, weights_only=False)
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
model = model.to(device)
model.eval()



########## Make submission ##########

##### DATA LOADER #####
test_df = pd.DataFrame({ 'ImageId': list_img_test, 'EncodedPixels': None })
ds = AirbusUnetDataset(test_df, mode='test')
loader = torch.utils.data.DataLoader(dataset=ds, shuffle=False, batch_size=2, num_workers=0)
    

##### EVALUATE #####
out_pred_rows = []

# When mode='test' the dataset returns img,label where label=img_file_name
for batch_num, (images, images_names) in enumerate(tqdm(loader, desc='Test', file=stdout)):
    # Evaluate test images
    outputs = model(images.to(device))

    for i, image_name in enumerate(images_names):
        mask = F.sigmoid(outputs[i,0]).data.detach().cpu().numpy()
        cur_seg = binary_opening(mask>0.5, disk(2))
        cur_rles = multi_rle_encode(cur_seg)
        
        if len(cur_rles)>0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle }]
        else:
            out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]
        

##### MAKE CSV #####
submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
res_path = os.path.join(out_path, 'submission.csv')
submission_df.to_csv(res_path, index=False)

print("Done. Created submission.csv")