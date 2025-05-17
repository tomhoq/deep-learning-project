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
    Returns dataframes with balanced classes (max tot images for each class, e.g. 1000 images with 0 ships, 1000 images with 1 ship, ...).
    Each row contains the ImageId, the rle mask and the number of ships for that image (Counts). 
    
    Returns:
        dict: { 'train': train_df, 'validation': valid_df }

    Example:
            ImageId                                      EncodedPixels     Counts
    0  000155de5.jpg  264661 17 265429 33 266197 33 266965 33 267733...      1
    1  000194a2d.jpg  360486 1 361252 4 362019 5 362785 8 363552 10 ...      5
    2  000194a2d.jpg  51834 9 52602 9 53370 9 54138 9 54906 9 55674 ...      5
    3  000194a2d.jpg  198320 10 199088 10 199856 10 200624 10 201392...      5
    4  000194a2d.jpg  55683 1 56451 1 57219 1 57987 1 58755 1 59523 ...      5
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
