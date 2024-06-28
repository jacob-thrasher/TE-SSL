import scipy
import numpy as np
from scipy.ndimage.interpolation import *
from PIL import Image
from skimage.transform import resize
import skimage
from torch import nn
import torch
import torchio.transforms as T
import random
from torchvision.transforms import v2
import random

class BatchedAugment(nn.Module):
    def __init__(self):
        super(BatchedAugment, self).__init__()
        sigma = np.random.uniform(0.0,1.0,1)[0]
        self.pipeline = T.Compose([
            T.RandomFlip(p=0.5),
            T.RandomNoise(),
            # T.RandomGhosting()
            T.RandomBlur(sigma)
            # T.RandomGamma()
        ])

    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
    
    def apply_augment(self, x, mode):
        if mode == 'train': 
            x = self.randomCrop(x, 96, 96, 96)
            return self.pipeline(x)

        else: return self.centerCrop(x, 96, 96, 96)


    def forward(self, batch, mode):
        return torch.stack([self.apply_augment(x, mode) for x in batch])




# Unused, but left just in case:

def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return shift(image, (int(offset[0]), int(offset[1]), int(offset[2])), order=order, mode='constant')


def scaleit(image, factor, isseg=False):
    order = 0 if isseg == True else 3

    height, width, depth= image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = depth

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = (height - zheight) // 2
        col     = (width - zwidth) // 2
        layer   = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2

        newimg = zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
        
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image

def resampleit(image, dims, isseg=False):
    order = 0 if isseg == True else 5

    image = zoom(image, np.array(dims)/np.array(image.shape, dtype=np.float32), order=order, mode='nearest')

    if image.shape[-1] == 3: #rgb image96, 
        return image if isseg else (image-image.min())/(image.max()-image.min()) 

def flipit(image):
    lr_thred = np.random.uniform(0,1,1)[0]
    ud_thred = np.random.uniform(0,1,1)[0]
    
    if lr_thred<=0.5:
        image = np.fliplr(image)
    if ud_thred>=0.5:
        image = np.flipud(image)
    
    return image


def intensifyit(image, factor):

    return image*float(factor)


def rotateit(image, axes, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return rotate(image, float(theta), axes=axes, reshape=False, order=order, mode='constant')

class CustomResize(object):
    def __init__(self, trg_size):

        self.trg_size = trg_size


    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img_array, trg_size):
        res = resize(img_array, trg_size)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res

class CustomToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            
            # backward compatibility
            return img.float()