from typing import Literal
import pandas as pd
import torch
from models.unet.src.utils.dataframe import get_dataframe
from utils.helpers import PATHS, masks_as_image
import os
import torchvision.transforms as transforms
from utils.data_augmentation import CenterCrop, DualCompose, HorizontalFlip, RandomCrop, RandomLighting, Rotate, VerticalFlip
from skimage.io import imread
import numpy as np


def get_unet_train_val_datasets(transform = None):
    if transform is None:
        train_transform, val_transform = get_transforms()

    df = get_dataframe()
    train_ds = AirbusUnetDataset(df['train'], transform = train_transform)
    valid_ds = AirbusUnetDataset(df['validation'], transform = val_transform)

    return train_ds, valid_ds


class AirbusUnetDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, mode: Literal['train', 'test'] = 'train'):
        # Get ImageIds and masks from the dataframe
        grp = list(df.groupby('ImageId'))
        self.image_ids =  [_id for _id, _ in grp] 
        self.image_masks = [m['EncodedPixels'].values for _,m in grp]

        self.transform = transform
        self.mode = mode
        
        # Baseline transform for the images
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]    # use mean and std from ImageNet 
        )


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        # Get current ImageId
        img_file_name = self.image_ids[idx]

        # Get the image either from the TRAIN folder or the TEST folder (depending on the mode)
        img_path = os.path.join(PATHS[self.mode], img_file_name)

        # Get the image and the mask (i.e. the ground truth)
        img = imread(img_path)
        mask = masks_as_image(self.image_masks[idx])
        
        # Apply the custom transform
        if self.transform is not None: 
            img, mask = self.transform(img, mask)

        if self.mode == 'train':
            label = torch.from_numpy(np.moveaxis(mask, -1, 0)).float()  
        else:
            label = str(img_file_name)

        return self.img_transform(img), label


def get_transforms():
    image_size = 384

    train_transform = DualCompose([
        Rotate(20),
        RandomLighting(0.05, 0.05),
        HorizontalFlip(),
        VerticalFlip(),
        RandomCrop((image_size,image_size,3))
    ])
    val_transform = DualCompose([CenterCrop((512,512,3))])

    return train_transform, val_transform