from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from utils.helpers import PATHS, rle2bbox
import pandas as pd


def sample_fn(x):
    """
    Keep max MAX_SAMPLES_NO_SHIPS of images with no ships and
    max MAX_SAMPLES of images with ships (for each class)
    """

    MAX_SAMPLES_NO_SHIPS = 1000
    MAX_SAMPLES = 25000

    # Undersample no ships images by 1000
    if (x.Counts == 0).all():
        return x.sample(MAX_SAMPLES_NO_SHIPS) 
    # Undersample images with ships (limit number to MAX_SAMPLES)
    else:
        return x.sample(min(len(x), MAX_SAMPLES)) 


def get_dataframe():
    """
    Returns dataframes with balanced classes (max 1000 images for each class, e.g. 1000 images with 0 ships, 1000 images with 1 ship, ...).
    Each row contains the ImageId, the Bbox (in yolo format: [xc, yc, h, w]) and the Bbox area and the number of ships for that image (Counts).
    
    Returns:
        dict: { 'train': train_df, 'validation': valid_df }

    Example:
            ImageId                                               Bbox  BboxArea  Counts
    0  000194a2d.jpg  [0.625, 0.38671875, 0.026041666666666668, 0.02...     440.0     5 
    1  000194a2d.jpg  [0.09830729166666667, 0.4967447916666667, 0.01...     153.0     5
    2  000194a2d.jpg  [0.3665364583333333, 0.23372395833333334, 0.01...     517.0     5
    3  000194a2d.jpg  [0.09765625, 0.5032552083333334, 0.00130208333...       6.0     5
    4  000194a2d.jpg  [0.4557291666666667, 0.244140625, 0.0247395833...     722.0     5
    """

    # Get all
    df = pd.read_csv(os.path.join(PATHS['root'], 'train_ship_segmentations_v2.csv'))

    ########## FETCHES AND CLEARS THE DATA ##########

    # Corrupted images
    exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']
    # Remove corrupted file
    df = df[~df['ImageId'].isin(exclude_list)]

    # Count number of ships for each image
    ship_count_df = df.copy()
    ship_count_df["Counts"] = ship_count_df["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
    ship_count_df = ship_count_df.groupby("ImageId").agg({ "Counts":"sum" }).reset_index()


    ########## MODIFY DF FOR YOLO (bboxes instead of mask rle) ##########

    IMG_SIZE = 768

    # Add bbox
    df["Bbox"] = df["EncodedPixels"].apply(lambda x: rle2bbox(x,(IMG_SIZE,IMG_SIZE)) if isinstance(x, str) else [])
    # Remove RLE
    df.drop("EncodedPixels", axis=1, inplace=True)

    # Add box area
    df["BboxArea"] = df["Bbox"].map(lambda x:x[2]*IMG_SIZE*x[3]*IMG_SIZE if len(x) > 0 else 0)

    # Remove boxes which are less than 1 percentile
    PERCENTILE_THRESHOLD = 1
    df = df[ df["BboxArea"] > np.percentile(df["BboxArea"], PERCENTILE_THRESHOLD) ]


    ########## BALANCE DATASET ##########
    # Keep max tot images for each class (e.g. 1000 images with 0 ships, 2000 images with 1 ship, ...)

    # Filter ImageIds based on the above critera (keep a max for each class)
    ship_count_df = ship_count_df.groupby("Counts")[['ImageId', 'Counts']].apply(sample_fn).reset_index(drop=True)

    # Filter the main df (that has bboxes, ...) to keep only the filtered ImageIds
    balanced_df = df.merge(ship_count_df[["ImageId"]], how="inner", on="ImageId")

    
    ########## SPLIT DATASET ##########

    # Split ImageIds into train and validation
    train_ids, val_ids = train_test_split(ship_count_df, test_size=0.05, stratify=ship_count_df['Counts'], random_state=42)

    # Filter the balanced_df into the two partitions
    train_df = pd.merge(balanced_df, train_ids)
    valid_df = pd.merge(balanced_df, val_ids)

    return {
        'train': train_df,
        'validation': valid_df,
    }
