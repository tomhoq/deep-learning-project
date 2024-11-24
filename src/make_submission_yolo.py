import os
from sys import argv, stdout
import numpy as np
from skimage.morphology import binary_opening, disk
import torch
import pandas as pd
from tqdm import tqdm
from models.unet.src.utils.dataset import AirbusUnetDataset
from utils.get_model import get_model
from utils.helpers import PATHS, multi_rle_encode
import torch.nn.functional as F
from PIL import Image, ImageDraw



########## Arguments ##########
# Check arguments
if len(argv) != 2:
    raise ValueError("Expected exactly two arguments. Usage: python evaluate.py <out_path>")

model_name = 'yolo'
out_path = argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_img_test = os.listdir(PATHS['test'])

print(f'''[*] Making submission:
        Model:   {model_name}
        Device:  {device.type}
        Samples: {len(list_img_test)}
''')


########## Load model ##########
model_path = os.path.join(out_path, 'model.pt')

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
# Keep the unet dataset because we're just using as a facade, the 'test' mode deoesn't do much
ds = AirbusUnetDataset(test_df, mode='test')
loader = torch.utils.data.DataLoader(dataset=ds, shuffle=False, batch_size=2, num_workers=0)
    


##### EVALUATE #####
out_pred_rows = []

def draw_bboxes_on_mask(boxes, image_size):
    """Draw bounding boxes on a binary mask."""
    mask = Image.new('1', image_size, 0)  # Binary mask
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)  # Convert to numpy array


# When mode='test' the dataset returns img, label where label=img_file_name
for batch_num, (images, images_names) in enumerate(tqdm(loader, desc='Test', file=stdout)):
    outputs = model(images.to(device))  # YOLO outputs bounding boxes and scores

    for i, image_name in enumerate(images_names):
        output = outputs[i]  # Extract output for a single image
        boxes = output[..., :4]  # Bounding boxes [x1, y1, x2, y2]
        scores = output[..., 4]  # Objectness scores

        # Filter predictions with a confidence threshold
        confidence_threshold = 0.5
        valid_indices = scores > confidence_threshold
        boxes = boxes[valid_indices]

        if len(boxes) > 0:
            # Draw bounding boxes on a binary mask
            mask = draw_bboxes_on_mask(boxes, images[i].shape[1:])
            
            # Encode the mask using RLE
            rle_encodings = multi_rle_encode(mask)
            
            # Append each RLE to the submission
            for rle in rle_encodings:
                out_pred_rows.append({'ImageId': image_name, 'EncodedPixels': rle})
        else:
            # If no detections, append a blank RLE
            out_pred_rows.append({'ImageId': image_name, 'EncodedPixels': None})


##### MAKE CSV #####
submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
res_path = os.path.join(out_path, 'submission.csv')
submission_df.to_csv(res_path, index=False)

print("[+] Done. Created submission.csv")