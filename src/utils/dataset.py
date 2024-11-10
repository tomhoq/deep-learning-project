import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from .helpers import masks_as_image
from .helpers import PATHS
from utils.data_augmentation import CenterCrop, DualCompose, HorizontalFlip, RandomCrop, VerticalFlip


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




def get_dataframes():
    """
    Make the dataframes for the dataset. Check `eda.ipynb` for a more detailed explanation.
    Basically we're returning a df that has been cleaned of corrupted images and images without ships.
    """

    # Get all
    masks = pd.read_csv(os.path.join(PATHS['root'], 'train_ship_segmentations_v2.csv'))
    # Remove corrupted file
    masks = masks[~masks['ImageId'].isin(['6384c3e78.jpg'])]
    # Remove images without ships
    masks = masks.dropna() 

    # Split between train and validation sets
    # We use stratify to balance the number of ships per image between the two df
    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.05, stratify=unique_img_ids['counts'], random_state=42)

    # Inner join masks with the ids
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, val_ids)

    # Update the counts, set to 0 all the cases where the mask is not a valid one (i.e. not a string)
    train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
    valid_df['counts'] = valid_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)

    return train_df, valid_df



def get_transforms():
    train_transform = DualCompose([HorizontalFlip(), VerticalFlip(), RandomCrop((256,256,3))])
    val_transform = DualCompose([CenterCrop((512,512,3))])

    return train_transform, val_transform