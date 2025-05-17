import random
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from PIL import Image
import torchvision.transforms as transforms



def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask

class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask

class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]

        return img, mask

class CenterCrop:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2
        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2,:]
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[y1:y2, x1:x2,:]

        return img, mask


class Resize:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, img, mask=None):
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.shape),
        ])

        img = t(img)
        if mask is not None:
           mask = t(mask) 

        return img, mask


class RandomLighting:
    def __init__(self, brightness=0.2, contrast=0.2):
        """
        Initializes the RandomLighting transform.

        Args:
            brightness (float): Maximum brightness adjustment factor (range [-b, b]).
            contrast (float): Maximum contrast adjustment factor (range [-c, c]).
        """
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img, mask=None):
        """
        Applies the random lighting transform to the given image.
        The mask remains unaltered.

        Args:
            img (PIL Image or Tensor): Input image to transform.
            mask (PIL Image, Tensor, or None): Input mask to return unaltered.

        Returns:
            tuple: Transformed image and unaltered mask.
        """
        # Generate random factors for brightness and contrast
        b_rand = torch.empty(1).uniform_(-self.brightness, self.brightness).item()
        c_rand = torch.empty(1).uniform_(-self.contrast, self.contrast).item()

        # Ensure the image is a PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Adjust brightness
        img = adjust_brightness(img, 1 + b_rand)

        # Adjust contrast
        if c_rand < 0:
            c_rand = -1 / (c_rand - 1)
        else:
            c_rand += 1
        img = adjust_contrast(img, c_rand)

        # Convert back to NumPy array
        img = np.array(img)

        # Return image and mask (mask remains unaltered)
        return img, mask
