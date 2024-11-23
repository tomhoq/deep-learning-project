from typing import Literal
import pandas as pd
import torch
from models.yolo.dataframe import get_dataframe
from utils.helpers import PATHS
import os
from PIL import Image
import torchvision.transforms as transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes




def get_yolo_train_val_datasets(transform = None):
    if transform is None:
        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    df = get_dataframe()
    train_ds = AirbusYOLODataset(df['train'], transform = transform)
    valid_ds = AirbusYOLODataset(df['validation'], transform = transform)

    return train_ds, valid_ds


class AirbusYOLODataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, mode: Literal['train', 'test'] = 'train', S = 7, B = 2, C = 1, transform = None):
        self.df = df
        self.files_dir = PATHS[mode]
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C


    def __len__(self):
        return len(self.df)


    def bboxes2cells(self, bboxes: torch.Tensor):
        """
        Convert bboxes to cells (if this image has to ships bboxes = [] => label_matrix will stay zeroed)
        """

        # We only have 1 class label (ship)
        class_label = 0

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in bboxes:
            if (len(box) == 0):
                continue

            x, y, height, width = box.tolist()

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (width * self.S, height * self.S)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, class_label+2:class_label+6] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return label_matrix


    def __getitem__(self, index):
        # Get ImageId
        img_id = self.df.iloc[index]['ImageId']

        # Get all the bboxes for the given ImageId
        group = self.df[self.df['ImageId'] == img_id].groupby('ImageId')
        bboxes = torch.tensor([el['Bbox'].values.tolist() for _, el in group][0])

        # Fetch and handle image
        img_path = os.path.join(self.files_dir, img_id)
        image = Image.open(img_path)
        image = image.convert("RGB")
        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        return image, self.bboxes2cells(bboxes)
