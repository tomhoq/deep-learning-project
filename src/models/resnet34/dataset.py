import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.helpers import PATHS
from utils.data_augmentation import CenterCrop, DualCompose, HorizontalFlip, RandomCrop, VerticalFlip


class AirbusDataset(Dataset):
    def __init__(self, mode='train'):

        in_df = _get_dataframes(mode)
        self.image_ids = list(in_df['ImageId'])
        self.y = list(in_df['has_ship'])

        self.transform = _get_transforms(mode)
        self.mode = mode


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]

        # Get the image either from the TRAIN folder or the TEST folder (depending on the mode)
        if (self.mode == 'train') | (self.mode == 'validation'):
            img_path = os.path.join(PATHS['train'], img_file_name)
        else:
            img_path = os.path.join(PATHS['test'], img_file_name)

        img = Image.open(img_path)
        img = self.transform(img)

        if (self.mode == 'train') | (self.mode == 'validation'):
            label = self.y[idx]
        else:
            label = 0
        
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return img, label




def _get_dataframes(mode):
    """
    Make the dataframes for the dataset. Check `eda.ipynb` for a more detailed explanation.
    Basically we're returning a df that has been cleaned of corrupted images.
    """

    #corrupted images
    exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

    # Get all
    masks = pd.read_csv(os.path.join(PATHS['root'], 'train_ship_segmentations_v2.csv'))
    # Remove corrupted file
    masks = masks[~masks['ImageId'].isin(exclude_list)]

    # Split between train and validation sets
    # We use stratify to balance the number of ships per image between the two df
    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.05, stratify=unique_img_ids['counts'], random_state=42)

    # Inner join masks with the ids
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, val_ids)

    # Set has_ship =  if ship is present, otherwise 0
    train_df['has_ship'] = train_df.apply(lambda c_row: 1 if isinstance(c_row['EncodedPixels'], str) else 0, 1)
    valid_df['has_ship'] = valid_df.apply(lambda c_row: 1 if isinstance(c_row['EncodedPixels'], str) else 0, 1)

    train_df = train_df.drop(columns=['EncodedPixels', 'counts'])
    valid_df = valid_df.drop(columns=['EncodedPixels', 'counts'])

    # Remove duplicate lines
    train_df = train_df.groupby('ImageId').agg({ 'has_ship': 'first' }).reset_index()
    valid_df = valid_df.groupby('ImageId').agg({ 'has_ship': 'first' }).reset_index()

    if mode == 'train':
        return train_df
    else:
        return valid_df


def _get_transforms(mode):
    img_dimensions = 384

    # Normalize to the ImageNet mean and standard deviation
    # Could calculate it for the cats/dogs data set, but the ImageNet
    # values give acceptable results here.
    train_transform = transforms.Compose([
        transforms.RandomRotation(50),
        transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((img_dimensions, img_dimensions)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_dimensions, img_dimensions)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    return train_transform if mode == 'train' else val_transform
