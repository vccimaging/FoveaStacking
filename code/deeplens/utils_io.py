import numpy as np
from scipy.signal import convolve2d
import cv2
import torch

def srgb2lin(img):
    """
    Convert an sRGB image to linear RGB (actual light intensity), range of image should be [0,1].
    
    Args:
        - img: a numpy or torch array of shape (H,W,3) representing an sRGB image
    
    Returns:
        - img: a numpy or torch array of shape (H,W,3) representing a linear RGB image
    """
    if isinstance(img,torch.Tensor):
        img = img.clone()
    elif isinstance(img,np.ndarray):
        img = img.copy()
    mask = img <= 0.04045
    img[mask] = img[mask] / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055)** 2.4
    return img

def lin2srgb(img):
    """
    Convert a linear RGB (actual light intensity) image to sRGB.
    
    Args:
        - img: a numpy or torch array of shape (H,W,3) representing a linear RGB image
    
    Returns:
        - img: a numpy or torch array of shape (H,W,3) representing an sRGB image
    """
    if isinstance(img,torch.Tensor):
        img = img.clone()
    elif isinstance(img,np.ndarray):
        img = img.copy()
    mask = img <= 0.0031308
    img[mask] = img[mask] * 12.92
    img[~mask] = 1.055 * (img[~mask]**(1/2.4)) - 0.055
    return img


def read_png_u8_RGB(file_name,normalize=True,read_alpha=False,color_space='srgb'):
    """
    read a png image and return it as a numpy array of shape (H,W,3) or (H,W,4) if read_alpha is True.
    opencv reads the image in BGR format, so the image is converted to RGB format.

    Args:
        - file_name: path to the png image
        - normalize: if True, the image is normalized to [0,1]
        - read_alpha: if True, the image is read as RGBA, otherwise it is read as RGB
        - color_space: 'linear' or 'srgb'. If 'linear', the image is converted to linear RGB, otherwise it is kept in sRGB
    
    Returns:
        - img: a numpy array of shape (H,W,3) or (H,W,4) if read_alpha is True
    """

    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED) # image is grayscale of shape (H,W)
    
    assert img.dtype == np.uint8, "Image should be of type np.uint8"
    if normalize:
        img = (img/255.0).astype(np.float32)
    
    if read_alpha:
        assert img.shape[-1] == 4, "Image should have 4 channels"
        img[...,:3] = img[...,:3:-1] # BGR to RGB
    else:
        img = img[...,:3]
        img = img[...,::-1] # BGR to RGB
    
    if color_space == 'linear':
        img = srgb2lin(img)
        
    return img.copy()

def read_tiff(file_name,normalize=True):
    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED) # image is grayscale of shape (H,W)
    if normalize:
        if img.dtype == np.uint16:
            img = (img/65535.0).astype(np.float32)
        elif img.dtype == np.uint8:
            img = (img/255.0).astype(np.float32)
        else:
            raise Exception("Unsupported image type")
    return img

def npy2tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img,0)
    elif img.ndim == 3:
        img = np.transpose(img,(2,0,1))
    else:
        raise Exception("Unsupported image shape")
    img = torch.tensor(img)
    return img

def read_img(file_name,normalize=True):
    if file_name.endswith(".png") or file_name.endswith('jpeg'):
        img = read_png_u8_RGB(file_name,normalize)
    elif file_name.endswith(".tiff"):
        img = read_tiff(file_name,normalize)
    else:
        raise Exception("Unsupported image format")
    return img

def to_uint8(img,enforce_range=True):
    # assert img.dtype == np.float32, "Image should be of type np.float32"
    if enforce_range:
        assert img.min() >= 0 and img.max() <= 1, "Image should be in range [0,1]"
    else:
        img = np.clip(img,0,1)
    return (img*255).astype(np.uint8)

def to_uint16(img,enforce_range=True):
    assert img.dtype == np.float32, "Image should be of type np.float32"
    if enforce_range:
        assert img.min() >= 0 and img.max() <= 1, "Image should be in range [0,1]"
    else:
        img = np.clip(img,0,1)
    return (img*65535).astype(np.uint16)

def demosaic(raw,Bayer_pattern='GBRG'):
    ''' Demosaic a raw image.
    
    Args:
    - raw: a 2D numpy array representing the raw image. with shape (H, W).
    - Bayer_pattern: the Bayer pattern of the raw image. Default is 'GBRG'. currently only support 'GBRG'.

    Returns:
    - color_img: a 3D numpy array representing the color image. with shape (H, W, 3).

    '''
    # Create an empty color image
    color_img = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=raw.dtype)

    kernel1 = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    kernel2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4

    if Bayer_pattern == 'GBRG':
        # Green channel
        color_img[0::2, 0::2, 1] = raw[0::2, 0::2]
        color_img[1::2, 1::2, 1] = raw[1::2, 1::2]

        # Blue channel
        color_img[0::2, 1::2, 2] = raw[0::2, 1::2]

        # Red channel
        color_img[1::2, 0::2, 0] = raw[1::2, 0::2]
    else:
        raise ValueError("Unknown Bayer pattern")

    # Interpolate the missing green pixels
    color_img[...,1] = convolve2d(color_img[...,1], kernel1, mode='same', boundary='symm')

    # Interpolate the missing red pixels
    color_img[..., 0] = convolve2d(color_img[..., 0], kernel2, mode='same', boundary='symm')

    # Interpolate the missing blue pixels
    color_img[..., 2] = convolve2d(color_img[..., 2], kernel2, mode='same', boundary='symm')

    color_img = np.clip(color_img, 0, 1)
    return color_img