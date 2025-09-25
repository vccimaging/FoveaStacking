# -*- coding: utf-8 -*-
"""
update the sharpness based focus stacking method using torch operations

"""

import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os, sys
import argparse
import glob
import torch 
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

from deeplens import LaplacianPyramidLoss
from torch.nn.functional import conv2d

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def plot_class(fig_dir,data,title, n_class):
    """
    Helper function to plot data with associated colormap and customized colorbar ticks.
    """
    newcolors = mpl.colormaps['hot'](np.linspace(0, 1, n_class))
    cmap = ListedColormap(newcolors)
    # fig, axs = plt.subplots(1, 1, figsize=(1 * 2 + 2, 3),
    #                         layout='constrained', squeeze=False)
    fig, ax = plt.subplots()
    # Plot the data
    data = data[::-1] # flip the data to make the top left corner the origin
    # Note that the column index corresponds to the x-coordinate, and the row index corresponds to y.
    ax.set_aspect('equal')
    psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=0, vmax=n_class)
    # Customize the colorbar
    cbar = fig.colorbar(psm, ax=ax)
    n_colors = cmap.N  # Number of discrete colors in the colormap
    ticks = np.linspace(0 + 0.5, n_class - 0.5, n_colors)  # Tick positions in the middle of each color
    tick_labels = np.arange(1, n_class+1)  # Integer labels for each color
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=16)  # Set font size for color bar
    ax.xaxis.set_visible(False)  # Hide X-axis
    ax.yaxis.set_visible(False)  # Hide Y-axis
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    plt.savefig(fig_dir, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def laplacian_kernel(size,channels=3, dtype=torch.float,dim="x"):  
    kernel = torch.zeros((size, size), dtype=dtype)  
    center = size // 2  
    if dim=="xy":
        for i in range(size):  
            for j in range(size):  
                kernel[i, j] = -1  
                if i == center and j == center:  
                    kernel[i, j] = size * size - 1  
    elif dim == "y":
        kernel[:,center] = -1
        kernel[center,center] = size-1
    elif dim == "x":
        kernel[center,:] = -1
        kernel[center,center] = size-1
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel 

def mean_kernel(size,channels,dtype=torch.float):
    kernel = torch.ones((size, size), dtype=dtype) / (size*size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def gaussian_kernel(size=5, channels=3, sigma=0, dtype=torch.float):
    if sigma == 0:
        sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8
    # Create Gaussian Kernel. In Numpy
    interval  = (2*sigma +1)/(size)
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    return kernel_tensor


class FocusStacker(object):
    def __init__(self, k_gauss=9, k_lap=7, k_blur=5,method="laplacian",fusion="max",lib="cv2"):
        self.method = method # choose between "sml" and "laplacian"
        self.fusion = fusion # choose between "max" and "mean"
        self.lib = lib # choose between "torch" and "cv2"
        assert method in ["sml","laplacian"], f"method: {method} not supported"
        assert fusion in ["max","mean"], f"fusion method: {fusion} not supported"
        assert lib in ["torch","cv2"], f"library: {lib} not supported"

        if lib == "torch":
            if method == "sml":
                self.gauss_kernel = gaussian_kernel(size=k_gauss,channels=1)
                self.laplacian_kernel_x = laplacian_kernel(size=k_lap,channels=1,dim="x")
                self.laplacian_kernel_y = laplacian_kernel(size=k_lap,channels=1,dim="y")
                self.mean_kernel = mean_kernel(size=k_blur,channels=1)
            elif method == "laplacian": 
                self.gauss_kernel = gaussian_kernel(size=k_gauss,channels=1)
                self.laplacian_kernel = laplacian_kernel(size=k_lap,channels=1,dim="xy")
                self.blur_kernel = gaussian_kernel(size=k_blur,channels=1)
        elif lib == "cv2":
            self.k_gauss = k_gauss 
            self.k_lap = k_lap
            self.k_blur = k_blur

    def sml(self, images):
        """
            Compute the sum of laplacian pyramid of a batch of images.
        """
        # compute the laplacian follow by gaussian blur
        images = images.mean(dim=1, keepdim=True)
        images = conv2d(images, self.gauss_kernel, padding="same")
        images_ml = conv2d(images, self.laplacian_kernel_x, padding="same").abs() + conv2d(images, self.laplacian_kernel_y, padding="same").abs()   # modified laplacian
        images_sml = conv2d(images_ml,self.mean_kernel,padding="same")  # average pool with k_lap
        print(f"max of images_sml:{images_sml.max()}")
        return images_sml

    def laplacian(self, images):
        """
            Compute the laplacian of a batch of images.
        """
        # compute the laplacian follow by gaussian blur
        images = images.mean(dim=1, keepdim=True)
        images = conv2d(images, self.gauss_kernel, padding="same")
        images = conv2d(images, self.laplacian_kernel, padding="same")
        images = images.abs()
        images = conv2d(images, self.blur_kernel, padding="same")
        return images


    def blur(self, images, kernel):
        """
            Compute the blur of a batch of images.
        """
        # compute the laplacian follow by gaussian blur
        images = images.mean(dim=1, keepdim=True)
        images = conv2d(images, self.gauss_kernel, padding="same")
        return images

    def fovea_stack(self, image_files, res=None):
        """
            Pipeline to focus stack a list of images.
        """

        img_list = [cv2.imread(img) for img in image_files]
        if res is not None:
            img_list = [cv2.resize(img, (res,res)) for img in img_list]
        if self.lib == "torch":
            images = torch.tensor(np.stack(img_list, axis=0)).permute(0,3,1,2).float() # NxCxHxW
            if self.method == "sml":
                sharpness = self.sml(images)
            elif self.method == "laplacian":
                sharpness = self.laplacian(images)

            sharpness = sharpness.permute(0,2,3,1).cpu().numpy() # NxHxWxC
            sharpness = sharpness.sum(axis=-1) # NxHxW

            # interpolate the reults in numpy
            images = images.permute(0,2,3,1).cpu().numpy() # NxHxWxC
        
        elif self.lib == "cv2":
            sharpness_list = []
            for img in img_list:
                if self.method == "laplacian":
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.GaussianBlur(img_gray, (self.k_gauss,self.k_gauss), 0)
                    img_lap = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=self.k_lap)
                    img_lap = np.abs(img_lap)
                    img_lap = cv2.GaussianBlur(img_lap, (self.k_blur,self.k_blur), 0)
                    sharpness_list.append(img_lap)
                else:
                    raise NotImplementedError("SML not implemented in cv2")
            sharpness = np.stack(sharpness_list, axis=0) # NxHxW
            images = np.stack(img_list, axis=0) # NxHxWxC
        
        print(f"shape of images: {images.shape}")

        print(f"shape of sharpness: {sharpness.shape}")
        logger.info("Using sharpness  to find regions of focus, and stack.")
        output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
        print(f"output shape: {output.shape}")

        if self.fusion == "max":
            maxima = sharpness.max(axis=0)
            bool_mask = np.array(sharpness == maxima)
            mask = bool_mask.astype(np.float32)/bool_mask.sum(axis=0)
        elif self.fusion == "mean":
            sharpness = (sharpness/sharpness.max(axis=0))**10
            mask = sharpness / sharpness.sum(axis=0) # soft attention mask

        print(f"mask shape: {mask.shape}")  

        # exp_lap = abs_laplacian+1e-6 # np.exp(abs_laplacian)
        # # exp_lap = np.exp(abs_laplacian.clip(max=100))
        # mask = exp_lap / exp_lap.sum(axis=0)

        print( mask.sum(axis=0).max(),mask.sum(axis=0).min())

        # for i, img in enumerate(images):
        #     output = cv2.bitwise_not(img, output, mask=mask[i])
        # output = 255 - output
        output = (images*mask[...,np.newaxis]).sum(axis=0)

        output.astype(np.uint8)

        return output,mask,sharpness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Focus stack images')
    parser.add_argument('--result_dir', type=str, default='data/camera_captured')
    parser.add_argument('--img_list', type=str, default='*.jpg')
    parser.add_argument('--k_gauss', type=int, default=19, help="kernel size of Gaussian Blurring used in pyramid")
    parser.add_argument('--k_lap', type=int, default=19, help="kernel size of Laplacian")
    parser.add_argument('--k_blur', type=int, default=127, help="kernel size of Gaussian Blurring")
    parser.add_argument('--method', type=str, default='laplacian', help="method to use for focus stacking")
    parser.add_argument('--fusion', type=str, default='mean', help="method to use for fusion")
    args = parser.parse_args()

    img_list = sorted(glob.glob(f"{args.result_dir}/{args.img_list}"))
    print(f"img_list: {img_list}")
    os.makedirs(f'{args.result_dir}/fovea_stack', exist_ok=True)
        

    stacker = FocusStacker(k_gauss=args.k_gauss, k_lap=args.k_lap, k_blur=args.k_blur,method=args.method,fusion=args.fusion)
    n_frames = len(img_list)

    
    output,mask,sharpness = stacker.fovea_stack(img_list)
    cv2.imwrite(f'{args.result_dir}/fovea_stack/stacked.png', output)
    
    # attention mask
    attention_mask = mask.argmax(axis=0)
    print(f"attention mask max: {attention_mask.max()}, min: {attention_mask.min()}")
    plot_class(f"{args.result_dir}/fovea_stack/attention_mask.png", attention_mask, title=f"", n_class=mask.shape[0])


    # save the lapacian
    for i in range(len(sharpness)):
        lap_i = np.abs(sharpness[i])
        lap_i = (lap_i - lap_i.min())/(np.max(lap_i)-lap_i.min())*255
        cv2.imwrite(f"{args.result_dir}/fovea_stack/sharpness_{i}.png", np.clip(lap_i,0,255))