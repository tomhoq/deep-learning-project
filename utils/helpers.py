import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

DATA_DIR = os.getenv('BLACKHOLE')
PATHS = {
    'root': DATA_DIR,
    'train': f"{DATA_DIR}/train_v2",
    'test': f"{DATA_DIR}/test_v2",
}


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    From https://www.kaggle.com/paulorzp/run-length-encode-and-decode    

    :param mask_rle: run-length as string formated (start length)
    :param shape: (height,width) of array to return 

    :returns: numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(rle_masks):
    """
    Convert masks from list (run-length encoded) to image

    :param rle_masks: An array of strings (where each string is the run-length encoded mask)
    """

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)

    for mask in rle_masks:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)

    return np.expand_dims(all_masks, -1)




#################### PLOTTING OR VIEWING IMAGES #################### 

def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """

    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]    
    return img


def imshow_tensor_with_mask(img, mask, title=None):
    """
    Imshow for Tensor.
    """

    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)
    fig = plt.figure(figsize = (6,6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 