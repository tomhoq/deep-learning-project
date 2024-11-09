import torch
import os
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from helpers import masks_as_image
from helpers import PATHS


class AirbusDataset(Dataset):
    def __init__(self, in_df, transform=None, mode='train'):
        grp = list(in_df.groupby('ImageId'))
        self.image_ids =  [_id for _id, _ in grp] 
        self.image_masks = [m['EncodedPixels'].values for _,m in grp]
        self.transform = transform
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]    # use mean and std from ImageNet 
        )


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]

        # Get the image either from the TRAIN folder or the TEST folder (depending on the mode)
        if (self.mode == 'train') | (self.mode == 'validation'):
            rgb_path = os.path.join(PATHS['train'], img_file_name)
        else:
            rgb_path = os.path.join(PATHS['test'], img_file_name)

        # Get the image and the mask (= ground truth)
        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[idx])
        
        if self.transform is not None: 
            img, mask = self.transform(img, mask)
            
        if (self.mode == 'train') | (self.mode == 'validation'):
            return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()  
        else:
            return self.img_transform(img), str(img_file_name)
