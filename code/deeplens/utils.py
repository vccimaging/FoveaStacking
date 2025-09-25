import os
import cv2 as cv
import random 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch.nn.functional as F
import logging
from datetime import datetime
from pytorch_msssim import SSIM

import yaml
import shutil
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from .basics import EPSILON

# ==================================
# PSF rotation and MTF related
# ==================================

    
def rot_pts(points, angles):
    """ Rotate N points by N different angles in radians.

    Args:
        points: shape [N, spp, 2]
        angles: shape [N], angles in radians
    Returns:
        points_rotated: shape [N, spp, 2]
    """
    # Create rotation matrices for each angle
    cos_angles = torch.cos(angles)  # shape [N]
    sinum_angles = torch.sin(angles)  # shape [N]
    rotation_matrices = torch.stack([
        torch.stack([cos_angles, -sinum_angles], dim=-1),  # shape [N, 2]
        torch.stack([sinum_angles, cos_angles], dim=-1)    # shape [N, 2]
    ], dim=-2)  # shape [N, 2, 2]

    # Perform batched matrix multiplication
    points_rotated = torch.einsum('ijk,ikl->ijl', points, rotation_matrices)  # shape [spp, N, 2]

    return points_rotated

def rotate_psf_map(points_shift, pointc_sensor, ra, ps ,ks=None):
        """ Rotate the PSF scattering points and integrate the PSF along sagittal and tangential axis.
        
        Args:
            points_shift (Tensor): Shape of [spp, N, 2, C] or [spp, N, 2], local point shift is in sensor plane.
            pointc_sensor (Tensor): Shape of [N, 2, C] or [N, 2]. position of the PSF center on the sensor plane.
            ra (Tensor): Shape of [spp, N , C] or [spp, N]. if the ray is valid, ra = 1, otherwise ra = 0.
            ps (float): pixel size in mm.
            ks (int): kernel size, if None, it will be calculated adaptively.
        Returns:
            psf (Tensor): Shape of [N, C, ks, ks] or [N, ks, ks]. PSF of the point source.
        """
        
        if len(points_shift.shape) == 3: # [spp, N, 2]
            # add a dim to make it [spp, N, 2, 1]
            points_shift = points_shift.unsqueeze(-1) # shape [spp, N, 2, 1]
            pointc_sensor = pointc_sensor.unsqueeze(-1) # shape [N, 2, 1]
            ra = ra.unsqueeze(-1) # shape [spp, N, 1]
            
        ra = ra.permute(1,0,2) # shape [N, spp, C]
        points_shift = points_shift.permute(1,0,2,3) # shape [N, spp, 2, C]
        N,spp,_,C = points_shift.shape
        
        
        angles = torch.arctan2(pointc_sensor[:,1].mean(dim=-1),pointc_sensor[:,0].mean(dim=-1)) # shape (N)
        if ks == None:
            rPSFs = torch.sqrt((points_shift.permute(0,1,3,2)[ra>0]**2).sum(dim=-1)) # shape (N, spp, C)
            ks = int(rPSFs.max().item()/ps)*2 + 1
            
        psfs_rotated = []
        for c in range(C):
            # ==> rotate the points and calculate PSF adaptively
            psf = rotate_psf(points_shift[...,c], ra[...,c], angles, ks, ps) # shape (grid**2, ks, ks)
            # normalize psf to 1
            psf = psf / psf.sum(dim=(-1,-2), keepdim=True) # shape (grid**2, ks, ks)
                             
            psfs_rotated.append(psf)

        psfs_rotated = torch.stack(psfs_rotated, dim=1) # shape (N, C, ks, ks)

        if C == 1:
            psfs_rotated = psfs_rotated.squeeze(1)
        
        return psfs_rotated.permute(0, 2, 3, 1)  # shape (N, ks, ks, C)
    
def rotate_psf(points_shift, ra, angles, ks, ps):
    """ Rotate the points by angle in radians and integrate the psf, the kernel size is adaptive given the pixel size.

    Args:
        points_shift: shape [N, spp, 2], or [spp, 2]
        ra: shape [N, spp], or [spp]
        angles: [N] angles in radians
        ks: kernel size
        ps: pixel size
    Returns:
        psf: shape [N, ks, ks]
    """
    from .monte_carlo import assign_pts_to_pixels

    # ==> Rotate the points
    points_shift_rotated = rot_pts(points_shift, angles) # shape [N, spp, 2]

    psf = assign_pts_to_pixels(points_shift_rotated.permute(1,0,2), ks, ps, ra.permute(1,0)) # assign points to pixels

    return psf


def psf2mtf(psf,ps):
    """ Convert 2D PSF kernel to MTF curve by FFT.

    Args:
        psf (tensor) [N,ks,ks]: 2D PSF kernel.
        ps (float): Pixel size in mm.

    Returns:
        freq (ndarray): Frequency axis.
        tangential_mtf (ndarray): Tangential MTF.
        sagittal_mtf (ndarray): Sagittal MTF.
    """
    N,ks,ks = psf.shape
    
    MTF2D = torch.abs(torch.fft.fft2(psf)) # (N, ks, ks)
    tangential_mtf = MTF2D[:, :, 0] # (N, ks)
    sagittal_mtf   = MTF2D[:, 0, :] # (N, ks)
    
    # Create frequency axis in cycles/mm
    freq = torch.fft.fftfreq(ks, ps)

    # Only keep the positive frequencies
    positive_freq_idx = freq >= 0

    return freq[positive_freq_idx], tangential_mtf[:,positive_freq_idx], sagittal_mtf[:,positive_freq_idx]


def RGBPSF2MTF(psf, ps):
    """
    calculate the MTF50 of the PSF.
    Args:
        psf: shape [N, ks, ks, 3], the PSF of the lens
        ps: pixel size
    Returns:
        MTF_dict: a dict containing the MTF50, frequency, and MTF curves
    """
    gray_psf = 0.2126 * psf[...,0] + 0.7152 * psf[...,1] + 0.0722 * psf[...,2]
    
    # Calculate tangent and sagittal MTFs
    freq, tan_mtf, sag_mtf = psf2mtf(gray_psf,ps)
    
    # Compute average MTF
    mtf = (tan_mtf + sag_mtf) / 2 # (N,ks//2)
    

    MTF_dict = {
        "freq": freq, # frequency in cycles/mm, shape (ks//2), starting from 0
        "mtf": mtf/ mtf[:,:1], # average MTF, shape (N, ks//2), decreasing
        "tan_mtf": tan_mtf / tan_mtf[:,:1], # tangential MTF, shape (N, ks//2), decreasing
        "sag_mtf": sag_mtf / sag_mtf[:,:1] # sagittal MTF, shape (N, ks//2), decreasing
    }
    
    return MTF_dict

def PSF2MTF(psf, ps):
    """
    calculate the MTF50 of the PSF.
    Args:
        psf: shape [N, ks, ks], the PSF of the lens
        ps: pixel size
    Returns:
        MTF_dict: a dict containing the MTF50, frequency, and MTF curves
    """
    # gray_psf = 0.2126 * psf[:,0] + 0.7152 * psf[:,1] + 0.0722 * psf[:,2]
    
    # Calculate tangent and sagittal MTFs
    freq, tan_mtf, sag_mtf = psf2mtf(psf,ps)
    
    # Compute average MTF
    mtf = (tan_mtf + sag_mtf) / 2 # (N,ks//2)
    

    MTF_dict = {
        "freq": freq, # frequency in cycles/mm, shape (ks//2), starting from 0
        "mtf": mtf/ mtf[:,:1], # average MTF, shape (N, ks//2), decreasing
        "tan_mtf": tan_mtf / tan_mtf[:,:1], # tangential MTF, shape (N, ks//2), decreasing
        "sag_mtf": sag_mtf / sag_mtf[:,:1] # sagittal MTF, shape (N, ks//2), decreasing
    }
    
    return MTF_dict

def calc_MTFnnP(mtf, freq,nn=50):
    """ get MTFnnP from mtf. This is corresponding to the freqency at nn% of the peak MTF.
    
    Args:
        mtf (np): MTF values of shape (N, ks)
        freq (np): frequency in cycles/mm of shape (ks,)
        nn (int): percent of MTF, 0-100 
    Returns:
        MTFnnP (tensor): MTFnnP in cycles/mm, of shape (N,)
    """
    # Normalize average MTF
    mtf /= mtf.max(dim=1,keepdim=True).values # normalized by peak value
    mtf = mtf.cpu().numpy()  # shape (N, ks)
    freq = freq.cpu().numpy()  # shape (ks,)
    mtfnn_list = []
    for i in range(len(mtf)):        
        mtfnn = np.interp(1-nn/100.0, mtf[i,::-1], freq[::-1])  # Interpolate using numpy.interp, mtf is decreasing, threfore inverse order is needed
        mtfnn_list.append(mtfnn)
    MTFnnP = torch.tensor(mtfnn_list)  # shape (N,)
    
    return MTFnnP

def calc_psf_stats(pts_shift, ra):
    """ Calculate the PSF statistics from the shifted points.
    Args:
        pts_shift: shape [spp, N, 2] or [spp, 2]
        ra: shape [spp, N] or [spp]
    Returns:
        pts_stats: dict containing the PSF statistics, including:
            - mse: mean square error of the PSF shift, shape [N]
            - rms: root mean square of the PSF shift, shape [N]
            - geo_mean: geometric mean of the PSF shift, shape [N]
            - mean: arithmetic mean of the PSF shift, shape [N]
            - raw: raw shifted points, shape [spp, N, 2]
            - ra: ray acceptance, shape [spp, N]
        pts_r: shape [spp, N], the valid PSF radius for each point.
    """
    N_ra = ra.sum(0) # shape [N] .add(EPSILON)
    # ==> Remove invalid pts
    pts_shift_valid = pts_shift*ra.unsqueeze(-1) # (spp, N, 2)
    
    pts_squares_valid = (pts_shift_valid**2).sum(-1) + EPSILON # (spp, N)
    pts_r = torch.sqrt(pts_squares_valid) # (spp, N) 
    
    N_ra = ra.sum(0) # shape [N]
    pts_mse = pts_squares_valid.sum(0) / N_ra # shape [N]
    pts_rms = torch.sqrt(pts_mse)
    pts_mean = pts_r.sum(0) / N_ra # shape [N]
    pts_geo_temp = pts_r.clone() # shape [spp, N]
    pts_geo_temp[ra<0.5] = 1
    pts_geo_mean = torch.exp(torch.log(pts_geo_temp).sum(0)/N_ra) # shape [N]

    pts_stats = {
        "mse": pts_mse, 
        "rms": pts_rms, 
        "geo_mean": pts_geo_mean, 
        "mean": pts_mean,
        "raw":pts_shift, 
        "ra":ra
        }
    
    return pts_stats, pts_r

def plot_spot_diagram(pts_shift, ra, pointc_sensor,fig_name=None,scale=1.0, **kwargs):
    """ Plot the spot diagram of the PSF shift, using different colors for each channel (RGB by default).
    
    Args:
        pts_shift: shape [spp, N, 2, C] or [spp, 2, C]
        ra: shape [spp, N, C] or [spp, C]
        pointc_sensor: shape [N, 2] or [2], the center of the PSF on the sensor plane
    """
    from matplotlib import cm
    spp, N, _, C = pts_shift.shape  # spp: samples per point, N: number of points, C: channels
    pts_shift = pts_shift*1000 # convert to um for plotting

    rms_list = []
    for c in range(C):
        pts_stats,pts_r = calc_psf_stats(pts_shift[...,c], ra[...,c]) # calculate the PSF statistics
        rms_list.append(pts_stats['rms'])  # shape [N]
    pts_rms = torch.stack(rms_list, dim=-1)  # shape [N, C]

    pts_c = pointc_sensor.detach().cpu().numpy()  # convert to numpy for plotting

    # Define colors for channels (RGB)
    channel_colors = ['r', 'g', 'b']
    if C > 3:
        # Extend colors if more than 3 channels
        channel_colors = list(cm.tab10.colors)[:C]

    # Gather all points from all channels and all N for global axis limits
    all_points = []
    for i in range(N):
        for c in range(C):
            pts = pts_shift[:, i, :, c][ra[:, i, c] > 0.5].detach().cpu().numpy()
            all_points.append(pts)
    all_points = np.concatenate(all_points, axis=0)
    x_min, x_max = all_points[..., 0].min() - 0.01, all_points[..., 0].max() + 0.01
    y_min, y_max = all_points[..., 1].min() - 0.01, all_points[..., 1].max() + 0.01
    # Make square axis
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)

    # Center the axis around the original point (0, 0)
    center_x, center_y = 0.0, 0.0
    half_range = max(abs(axis_max - center_x), abs(axis_max - center_y), abs(center_x - axis_min), abs(center_y - axis_min)) * scale
    axis_min = center_x - half_range
    axis_max = center_x + half_range

    if kwargs.get('layout'):
        if kwargs['layout'] == 'vertical':
            fig, axs = plt.subplots(N, 1, figsize=(3, 3 * N))

    else:
        fig, axs = plt.subplots(1, N, figsize=(3 * N, 3))
    if N == 1:
        axs = [axs]
    for i in range(N):
        ax = axs[i]
        for c in range(C):
            points = pts_shift[:, i, :, c][ra[:, i, c] > 0.5].detach().cpu().numpy()
            ax.scatter(points[..., 0], points[..., 1], s=0.2, color=channel_colors[c], label=f'Channel {c+1}')
            # # plot PSF center for each channel
            # if pts_c.ndim == 2:
            #     ax.plot(pts_c[i, 0], pts_c[i, 1], 'o', color=channel_colors[c], markersize=3)
            # else:
            #     ax.plot(pts_c[0], pts_c[1], 'o', color=channel_colors[c], markersize=3)
        ax.set_aspect('equal', adjustable='box')
        if kwargs.get('title'):
            if kwargs['title']!='off':
                ax.set_title(f"{kwargs['title']} with rms:{pts_rms.mean(dim=-1)[i]:.2f} (um)")
        else:
            ax.set_title(f'Spot Diagram {i+1} rms:{pts_rms.mean(dim=-1)[i]:.2f} (um)')
    
        if kwargs.get('xlim'):
            ax.set_xlim(kwargs['xlim'])
        else:
            ax.set_xlim([axis_min, axis_max])
        if kwargs.get('ylim'):
            ax.set_ylim(kwargs['ylim'])
        else:
            ax.set_ylim([axis_min, axis_max])
    
        # axis off optional
        if kwargs.get('axis_off', False):
            ax.axis('off')
        
    fig.tight_layout()
    if fig_name is None:
        plt.show()
    else:
        fig.savefig(f"{fig_name}.png")
        plt.close(fig)

def plot_pts_shift(pts_shift, ra, ks, ps, fig_name='temp/pts_shift'):
    """ Plot the PSF shift, for debugging purpose.
    
    Args:
        pts_shift: shape [spp, N, 2] or [spp, 2]
        ra: shape [spp, N] or [spp]
        ks: kernel size in number of pixels
        ps: pixel size in mm
    """
    spp,N,_ = pts_shift.shape # spp: number of samples per point, N: number of points
    # ==> Remove invalid pts
    pts_stats,pts_r = calc_psf_stats(pts_shift, ra) # calculate the PSF statistics
    pts_shift = pts_stats['raw'] # shape [spp, N, 2
    pts_mean = pts_stats['mean'] # shape [N]
    pts_rms = pts_stats['rms'] # shape [N]
    pts_geo_mean = pts_stats['geo_mean'] # shape [N]


    psf_range = [(- ks / 2 + 0.5) * ps, (ks / 2 - 0.5) * ps]    # this ensures the pixel size does not change in assign_pts_to_pixels function
    for i in range(N):
        points = pts_shift[:,i,:][ra[:,i]>0.5].detach().cpu().numpy()
        x_min, x_max = min(points[...,0].min(),psf_range[0])-0.01, max(points[...,0].max(),psf_range[1])+0.01
        y_min, y_max = min(points[...,1].min(),psf_range[0])-0.01, max(points[...,1].max(),psf_range[1])+0.01
        plt.close('all')  # close all previous plots
        
        # scatter plot
        fig, ax = plt.subplots()
        ax.scatter(points[..., 0], points[..., 1],s=0.5)
        # plot a box of the psf_range
        ax.plot([psf_range[0], psf_range[0], psf_range[1], psf_range[1], psf_range[0]],
            [psf_range[0], psf_range[1], psf_range[1], psf_range[0], psf_range[0]], 'r')
        # set x,y limit
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        fig.savefig(f'{fig_name}_{i:02d}.png')
        plt.close(fig)

        # plot heatmap of the points
        fig, ax = plt.subplots()
        h = ax.hist2d(points[..., 0], points[..., 1], bins=100, range=[[x_min, x_max], [y_min, y_max]])
        fig.colorbar(h[3], ax=ax)
        # plot a box of the psf_range
        ax.plot([psf_range[0], psf_range[0], psf_range[1], psf_range[1], psf_range[0]],
            [psf_range[0], psf_range[1], psf_range[1], psf_range[0], psf_range[0]], 'r')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        fig.savefig(f'{fig_name}_heatmap_{i:02d}.png')
        plt.close(fig)

        # plot histogram of pts_r
        fig, ax = plt.subplots()
        ax.hist(pts_r[:, i][ra[:, i] > 0.5].flatten().cpu().numpy(), bins=100)
        rms, mean, geo_mean = pts_rms[i].cpu().numpy(), pts_mean[i].cpu().numpy(), pts_geo_mean[i].cpu().numpy()
        # vertical lines for the point statistics
        ax.axvline(x=rms, color='r', linestyle='--')
        ax.axvline(x=geo_mean, color='orange', linestyle='--')
        ax.axvline(x=mean, color='lime', linestyle='--')
        ax.legend([f'rms:{rms:.3f}', f'geo_mean:{geo_mean:.3f}', f'mean:{mean:.3f}'])
        fig.savefig(f'{fig_name}_hist_{i:02d}.png')
        plt.close(fig)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
        
# ==================================
# Image batch quality evaluation
# ==================================

def batch_PSNR(img_clean, img, batch_mean = True):
    """ Compute PSNR for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    PSNR_list = []
    for i in range(Img.shape[0]):
        PSNR_list.append(torch.tensor(compare_psnr(Img_clean[i,:,:,:], Img[i,:,:,:])))
    PSNR = torch.stack(PSNR_list)
    if batch_mean:
        PSNR = PSNR.mean()
    return PSNR

def batch_SSIM(img, img_clean, multichannel=True, batch_mean = True):
    """ Compute SSIM for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255)
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255)
    ssim_module = SSIM(data_range=255, size_average=batch_mean, channel=3)
    SSIM_py = ssim_module(Img, Img_clean)
    return SSIM_py


# ==================================
# Image batch normalization
# ==================================

def normalize_ImageNet_stats(batch):
    """ Normalize dataset by ImageNet(real scene images) distribution. 
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    batch_out = (batch - mean) / std
    return batch_out


def de_normalize(batch):
    """ Convert normalized images to original images to compute PSNR.
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    batch_out = batch * std + mean
    return batch_out


# ==================================
def gpu_init(gpu=0):
    """Initialize device and data type.

    Returns:
        device: which device to use.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def set_logger(dir='./'):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')

    fhlr = logging.FileHandler(f"{dir}/output.log")
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')

    # fhlr2 = logging.FileHandler(f"{dir}/error.log")
    # fhlr2.setFormatter(formatter)
    # fhlr2.setLevel('WARNING')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    # logger.addHandler(fhlr2)


# ==================================
# Image processing
# ==================================

def estimate_H(ref_img, cap_img):
    # ==> Estimate homography matrix
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(cap_img, None)
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        machesMask = mask.ravel().tolist()

        h,w = ref_img.shape[:2]
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, H)

        cap_img = cv.polylines(cap_img, [np.int32(dst)], True, [0,0,255], 3, cv.LINE_AA)

        # draw inliers
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=machesMask,
                           flags=2)
        img3 = cv.drawMatches(ref_img, kp1, cap_img, kp2, good, None, **draw_params)
        
    else:
        raise Exception("Not enough matches are found - %d/%d" % (len(good), 4))
    
    return H,img3

def get_H(ref_img,cap_img):
    '''
        Estimate homography matrix H from ref_img to cap_img, and inv_H from cap_img to ref_img

        Args:
        - ref_img: a 2D numpy array representing the reference image. with shape (H, W, 3) uint8.
        - cap_img: a 2D numpy array representing the captured image. with shape (H, W, 3) uint8.

        Returns:
        - H: a 3x3 numpy array representing the homography matrix from ref_img to cap_img. (matrix assumes image's xy range is [-1,1] in pytorch convention) 
        - inv_H: a 3x3 numpy array representing the homography matrix from cap_img to ref_img. (matrix assumes image's xy range is [-1,1] in pytorch convention) 
    '''
    inv_H,match_img = estimate_H(cap_img,ref_img)
    H = np.linalg.inv(inv_H)
    # H,match_img = estimate_H(ref_img,cap_img)
    warped_img = cv.warpPerspective(ref_img, H, (cap_img.shape[1], cap_img.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=(255,255,255))
    # warped_rgb = cv.warpPerspective(ref_img, H, (cap_img.shape[1], cap_img.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=(255,255,255))
    
    # save images
    alpha=0.5
    save_imgs ={}
    scale =  warped_img.mean() / cap_img.mean()
    cap_img = (cap_img * scale).clip(0,255).astype(np.uint8)
    save_imgs["ref_warped"] = warped_img
    save_imgs["matches"] = match_img
    save_imgs[f"composit_alpha{alpha:.2f}"] = warped_img * alpha + cap_img * (1 - alpha)
    
    composit_RG = np.zeros_like(warped_img)
    composit_RG[:,:,1] = warped_img.mean(axis=-1) 
    composit_RG[:,:,2] = cap_img.mean(axis=-1)
    save_imgs[f"composit_RG"] = composit_RG
    print(f"H in pixel:\n{H}")

    # transform H to make bothe images in the range of [-1,1]
    inv_H = np.linalg.pinv(H)
    h_ref,w_ref = ref_img.shape[:2]
    h_cap,w_cap = cap_img.shape[:2]
    # torch affine_grid assume image is in [-1,1] range, so we need to normalize the matrix
    Ar = construct_affine_matrix(w_cap/2,h_cap/2,w_cap/2,h_cap/2)
    Al = construct_affine_matrix(w_ref/2,h_ref/2,w_ref/2,h_ref/2)
    Al = np.linalg.inv(Al)
    inv_H = np.matmul(Al, np.matmul(inv_H, Ar)).astype(np.float32)
    H =  np.linalg.inv(inv_H)
    

    return H,inv_H, save_imgs


def construct_affine_matrix(scale_x, scale_y, tx, ty):
    ''' construct affine matrix from scale, translation parameters'''
    A = np.array([[scale_x, 0, tx], [0, scale_y, ty], [0, 0, 1]])
    return A



# ==================================
# Configurations
# ==================================

def config(file_path='configs/simulate_optimize.yml', EXP_NAME='Simulate_opt',result_dir=None):
    # ==> Config
    with open(file_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    config_args(args, EXP_NAME, result_dir=result_dir)

    # ==> save the config, lens
    config_path = os.path.join(args['result_dir'], 'configs')
    os.makedirs(config_path, exist_ok=True)
    shutil.copy(file_path, os.path.join(config_path, os.path.basename(file_path))) # save config

    return args

def config_args(args, EXP_NAME='Simulate_opt',result_dir=None):
    # ==> Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # device= "cpu"
    args['device'] = device
    if EXP_NAME is None:
        assert 'EXP_NAME' in args, "Please provide EXP_NAME in args or as a parameter."
        EXP_NAME = args['EXP_NAME']

    # ==> Result folder
    # characters = string.ascii_letters + string.digits
    # random_string = ''.join(random.choice(characters) for i in range(4))
    if result_dir is None:
        result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + f'-{EXP_NAME}'
    args['result_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    print(f'Result folder: {result_dir}')

    lens_path = os.path.join(result_dir, 'lenses')
    os.makedirs(lens_path, exist_ok=True)
    lens_file = args['lens']['path']
    shutil.copy(lens_file, os.path.join(lens_path, os.path.basename(lens_file))) # save lens
    # ==> Logger
    set_logger(result_dir)
    logging.info(args)

    # ==> Random seed
    set_seed(args['train']['seed'])
    torch.set_default_dtype(torch.float32)

    return args



# ==================================
# Image visualization
# ==================================

def plot_scatter(x_, y_, value_,title,fig_name=None,radius=1.0,
                 vmin=None,vmax=None,
                 fix_bound=True,
                 cmap='coolwarm',
                 point_size=1,
                 colorbar=True,
                 axis_on=True,
                 figsize=(6,5)):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.scatter(x_,y_,c=value_,cmap=cmap,s=point_size,vmin=vmin,vmax=vmax)
    # cax = ax.imshow(opl,cmap='viridis',vmin = opl[mask].min(),vmax = opl[mask].max())
    ax.axis('off')
    if axis_on:
        ax.axis('on')
    
    if fix_bound:
        ax.set_xlim(-radius,radius)
        ax.set_ylim(-radius,radius)
    if colorbar:
        fig.colorbar(cax,ax=ax)
    ax.set_title(title)
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name) #, bbox_inches='tight')
        fig.clf()
        plt.close(fig)

# Function to plot difference image
def plot_difference(ax, data, title, cmap, max_abs=None):
    """ Plot difference of two RGB images."""
    im = ax.imshow(data, cmap=cmap, interpolation='nearest')
    if max_abs is None:
        max_abs = np.max(np.abs(data))
    im.set_clim(vmin=-max_abs, vmax=max_abs)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difference', rotation=270, labelpad=15)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_zern_coeffs(params,label=['zern',], fig_name=None, title='Zern. Amp. in [um]'):
    """ Plot Zernike coefficients.

        Parameters:
            params (torch.Tensor): Zernike coefficients, shape [K] or [N,K]
            label (list): List of labels for each coefficient, length should match params.shape[0] if params is 2D.
            fig_name (str): Path to save the plot.
            title (str): Title of the plot.
    """
    # Plot Zernike coefficients with controlled figure size
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(params.shape) == 1:
        params = params.unsqueeze(0)  # make it [1, K] for consistent plotting
    N, K = params.shape
    fig, ax = plt.subplots(figsize=(6,4))
    width = min(0.4, 1.0 / (N+1) )  # width of each bar, ensure it fits in the plot
    for i in range(N):
        ax.bar(np.arange(K) + i*width, params[i].detach().cpu() * 1000, label=label[i], color=colors[i], width=width)
    ax.set_title(title)
    ax.hlines(y=-2, xmin=0, xmax=K, colors='r', linestyles='dashed')
    ax.hlines(y=2, xmin=0, xmax=K, colors='r', linestyles='dashed')
    ax.legend()
    fig.tight_layout()
    if fig_name is None:
        plt.show()
    else:
        fig.savefig(fig_name)
        plt.close(fig)

def img_dif_RGB(img1, img2, dst_fig_name='diff.png'):
    """ Calculate and visualize the difference between two images in RGB channels.
    """
    # read images
    img1 = cv.imread(img1).astype(np.float32)
    img2 = cv.imread(img2).astype(np.float32)
    # calculate the difference
    diff = img1 - img2
    diff = diff[:, :, ::-1] # convert BGR to RGB

    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Red channel differences
    plot_difference(axs[0], diff[:,:,0], 'Red Channel Differences', 'RdBu_r',max_abs=40)

    # Plot Green channel differences
    plot_difference(axs[1], diff[:,:,1], 'Green Channel Differences', 'RdBu_r',max_abs=40)

    # Plot Blue channel differences
    plot_difference(axs[2], diff[:,:,2], 'Blue Channel Differences', 'RdBu_r',max_abs=40)

    # Adjust layout and display
    plt.tight_layout()

    # plt.axis('off')
    fig.savefig(dst_fig_name)
    plt.clf()

@torch.no_grad()
def draw_psf_with_center(
                    psfs,
                    ptc_sensor,
                    sensor_res,  # sensor resolution in pixels
                    sensor_size,  # sensor size in mm
                    log_scale=False,
                    fig_name='./psf.png',
                    ruler_len=500,
                    text_height=30,
                    fontsize=20,
                    ):
    """ Draw RGB PSF map at a certain depth. Will draw M x M PSFs, each of size ks x ks.
    
    Notice: the psfs by default is from bottom-left to top-right, which is different from the sensor coordinate.
    Args:
        psfs: [N, ks, ks, C] tensor
        ptc_sensor: [N, 2, C] tensor
        log_scale: whether to log scale the PSF
        fig_name: name of the saved figure
        ruler_len: length of the scale ruler in um
        text_height: height of the text
        fontsize: font size of the text
        
    """
    pixel_size = sensor_size[0] / sensor_res[0]  # pixel size in mm
    assert pixel_size == sensor_size[1] / sensor_res[1], "Pixel size should be the same in both dimensions."
    
    assert psfs.ndim == 4, f'psfs should be a 4D tensor, but got {psfs.ndim}D tensor.'
    N, ks, ks, C = psfs.shape
    # Los scale the PSF for better visualization
    if log_scale:
        psfs = torch.log(psfs + 1e-3)   # 1e-3 is an empirical value
    

    # Save figure using matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.invert_yaxis()  # Invert y-axis so it points down

    ptc_sensor = ptc_sensor.mean(dim=-1)  # average over color channels [N, 2]
    ptsc_pix = cvt_img2pix(ptc_sensor, sensor_res, sensor_size) # convert to pixel space
    pts_pix_TL = ptsc_pix.floor() - ks//2 # top-left corner of the PSF in pixel space
    pts_pix_TL = pts_pix_TL.cpu().numpy()

    # reshape psf_imgs to be plotted in a grid
    psf_img = psfs # [N, ks, ks, 3]
    v_min, v_max = psf_img.amin(dim=(1,2,3), keepdim=True), psf_img.amax(dim=(1,2,3), keepdim=True) 
    # psf_img = psf_img / psf_img.amax(dim=(1,2,3), keepdim=True) # normalize each PSF
    psf_img = (psf_img - v_min) / (v_max - v_min) # normalize each PSF to [0, 1]
    alpha = torch.ones_like(psf_img[...,:1])
    alpha[psf_img.sum(dim=-1, keepdim=True) == 0] = 0 # set alpha to 0 if pixel value is all 0
    psf_img = torch.cat((psf_img, alpha), dim=-1).cpu().numpy() # add alpha channel of shape [N, ks, ks, 4]
    for i in range(N):
        psf_i = psf_img[i] # convert to numpy array
        tx, ty = pts_pix_TL[i] # top-left corner of the PSF in pixel space
        # tx, ty = ptc_sensor[i].cpu().numpy()/ self.pixel_size # convert to pixel
        # tx, ty = int(tx) - ks//2 + W//2, int(ty) - ks//2 + H//2
        ax.imshow(psf_i, extent=(tx, tx + ks, ty, ty + ks),vmax=1,vmin=0,origin='lower')
        # print(f'Drawing PSF {i+1}/{N} at ({tx}, {ty}), extends=({tx}, {tx + ks}, {ty}, {ty + ks})')
        # break

    x_min,x_max = pts_pix_TL[...,0].min(), pts_pix_TL[...,0].max() + ks
    y_min,y_max = pts_pix_TL[...,1].min(), pts_pix_TL[...,1].max() + ks
    print(f'PSF grid extends: x=({x_min}, {x_max}), y=({y_min}, {y_max})')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min) 

    # draw the scale ruler
    arrow_end = ruler_len / (self.pixel_size * 1e3)   # plot a scale ruler
    plt.annotate('', xy=(x_min, y_max - text_height ), xytext=(x_min+arrow_end, y_max - text_height ), arrowprops=dict(arrowstyle='<->', color='white', lw=fontsize/8))
    plt.text(x_min + arrow_end + text_height, y_max - text_height, f'{ruler_len} um', color='white', fontsize=fontsize, ha='left')

    
    # set x,y ticks font size
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlabel('x [pixel]', fontsize=fontsize)
    ax.set_ylabel('y [pixel]', fontsize=fontsize)
    
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, dpi=300)
        plt.close()
    plt.style.use('default')

def calc_wavefront_aberration(lens,pt_rel,aper_idx = None, spp=64):
    """ 
        Calculate the delta OPD for the given relative position in the image. 
        
        Notice for the precise calculation, function should be called using torch.set_default_dtype(torch.float64)

    Args:
        lens (Lensgroup): the lens group object
        pt_rel (torch.Tensor): relative position in the image, shape [2] or [1,2] for u,v coordinates, range [-1,1] for both u and v
        aper_idx (int): the index of the aperture to use, default is None
        spp (int): number of samples per point, default is 64, which is the number of rays to sample in the pupil plane
        
    Returns:
        opd (torch.Tensor): optical path difference in the exit pupil plane, shape [N]
        pts_pupil_rel (torch.Tensor): relative position in the exit pupil plane, shape [N,2] in range [-1,1]
    """
    device = lens.device
    with torch.no_grad():
        # zern_val = lens.surfaces[lens.dpp_idx].get_zern_amp()
        # lens.surfaces[lens.dpp_idx].set_zern_amp([0])
        # 1. Backward Pass: sample ray on the sensor and trace to the object plane, close aperture to zero to consider the reference ray
        pt_img = cvt_rel2img(pt_rel.unsqueeze(0), lens.sensor_size) # convert relative position to image position
        pt_sensor = torch.cat([pt_img, torch.tensor([[lens.d_sensor]]).to(device)], dim=-1) # add a zero z-coordinate for the sensor plane, shape [1,3]
        ray = lens.sample_ray_points2pupil(pt_sensor, pupil="exit", spp=1, pupil_scale=0.0) # chief ray with zero aperture
        ray_obj = lens.trace2obj(ray, depth = lens.obj_plane.d) # trace the ray to the object plane
        pt_obj = ray_obj.o[0] # point in the object plane # [1,1,3] -> [1,3]

        # 2. Forward Pass: get ray tracing from object plane to sensor plane, fully open aperture to get aberrations
        ray = lens.sample_ray_points2pupil(pt_obj, pupil="entrance", spp=spp, pupil_scale=1.0) # fully open aperture
        # lens.surfaces[lens.dpp_idx].set_zern_amp(zern_val) # restore the zernike coefficients
        
    ray.coherent = True # set coherent to True to accumulate the OPL
    ray = lens.trace2sensor(ray) # trace the ray to the sensor plane

    # 3. Backward: free space propagate to exit pupil and fit zernike coefficients
    if aper_idx is None:
        aper_idx = lens.dpp_idx
    pupilz, pupilr = lens.exit_pupil(aper_idx=aper_idx) # get the exit pupil z-coordinate and radius
    ray = ray.propagate_to(pupilz)
    pts_pupil = ray.o[ray.ra>0] # points in the exit pupil plane, shape [N_valid,3]
    opl = ray.opl[ray.ra>0] # optical path length in the exit pupil plane, shape [N_valid]
    
    # 4. Fit Zernike coefficients
    opl_ref = torch.norm(pts_pupil - pt_sensor,dim=-1)
    opd = (opl + opl_ref)
    opd -= opd.mean()  # subtract the mean to center the OPD
    pts_pupil_rel = pts_pupil[:,:2]/pupilr
    

    return opd, pts_pupil_rel

@torch.no_grad()
def draw_psf_compact(psfs,log_scale=False, fig_name='./psf.png'):
    """ Draw RGB PSF map at a certain depth. Will draw M x M PSFs, each of size ks x ks.
    Notice: the psfs by default is from bottom-left to top-right, which is different from the sensor coordinate.
    """
    N,ks,ks,C = psfs.shape
    if log_scale:
        psfs = torch.log(psfs+1e-3)
    
    M = int(np.sqrt(N))
    psfs = psfs.reshape(M,M,ks,ks,C).cpu().numpy()
    # psfs = psfs[::-1,::-1] # flip the psfs to match the sensor coordinate
    # reshape psfs to Mxks, Mxks, 3
    psf_grid = np.zeros((M*ks,M*ks,3))
    for i in range(M):
        for j in range(M):
            v_max,v_min = psfs[i,j].max(), psfs[i,j].min()
            psf = (psfs[i,j]- v_min)/(v_max - v_min) # normalize each PSF to [0, 1]
                # 1e-3 is an empirical value
            psf_grid[i*ks:(i+1)*ks,j*ks:(j+1)*ks] = psf
    
    plt.figure(figsize=(10, 10))
    plt.imshow(psf_grid)
    plt.axis('off')
    plt.tight_layout(pad=0)
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, dpi=300)
        plt.clf()

def draw_val_2D(val,title,vmin=None,vmax=None,fig_name=None,cmap='plasma',colorbar=True):
    """ visualize value as colormapped 2D image
    """ 
    fig, ax = plt.subplots()
    ax.set_title(title,fontsize=16)
    im = ax.imshow(val,cmap=cmap,vmin=vmin,vmax=vmax)
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.axis('off')    
    if fig_name is None:
        plt.show()
    else:
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        fig.clf()
        plt.close(fig)

def plot_val_hist(val_list,label=['val'], title='Value Histogram',xlabel="Value",ylabel="Density", fig_name=None, bins=100):
    """ Plot histogram of values.
    
    Args:
        val_list (list): List of values to plot, can be a 1D tensor or a list of tensors.
        title (str): Title of the plot.
        fig_name (str): Path to save the figure. If None, will show the plot.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(4,3))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,val in enumerate(val_list):
        plt.hist(val.flatten().cpu().numpy(), bins=bins, label=label[i], color=colors[i],  alpha=0.7, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
        plt.clf()
        plt.close()

def plot_val_cdf(val_list,label=['val'], title='Value CDF',xlabel="Value",ylabel="Cumulative Probability", fig_name=None):
    """ Plot CDF of values.
    
    Args:
        val_list (list): List of values to plot, can be a 1D tensor or a list of tensors.
        title (str): Title of the plot.
        fig_name (str): Path to save the figure. If None, will show the plot.
    """
    plt.figure(figsize=(4,3))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,val in enumerate(val_list):
        val = val.flatten().cpu().numpy()
        val = np.sort(val)
        cdf = np.arange(1, len(val) + 1) / len(val)
        plt.plot(val, cdf,label=label[i], color=colors[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
        plt.clf()
        plt.close()
    

def draw_scatter(x_, y_, value_,title,fig_name=None,radius=1.0,vmin=None,vmax=None,fix_bound=True):
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.scatter(x_,y_,c=value_,cmap='coolwarm',s=1,vmin=vmin,vmax=vmax)
    # cax = ax.imshow(opl,cmap='viridis',vmin = opl[mask].min(),vmax = opl[mask].max())
    ax.axis('off')
    if fix_bound:
        ax.set_xlim(-radius,radius)
        ax.set_ylim(-radius,radius)
    # fig.colorbar(cax,ax=ax)
    # set color bar with ticks
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=20)  # Set font size for color bar
    ax.set_title(title,size=30)
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches='tight',transparent=False,dpi=300)
        plt.clf()
        plt.close()

def plot_mtfs(mtf_list,FoV=9.2):
    # define colormap as coolwarm
    cmap = plt.get_cmap('coolwarm')
    
    fig,ax = plt.subplots(figsize=(4, 3))
    
    for dict in mtf_list:
        freq = dict['freq']
        radial_mtf = dict['mtf']
        radius = dict['radius'] # within range [0,1]
        
        # plot the MTF
        color = cmap(radius)  # Get color from colormap based on radius
        ax.plot(freq, radial_mtf, color=color)
        # ax.plot(freq, radial_mtf, color=color, label=f'r={radius:.2f}')
    
    # set legend
    # ax.legend()
    # color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('FoV', rotation=270, labelpad=15)
    # set cbar tick
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'$0\degree$', f'${FoV/2:.1f}\degree$', f'${FoV:.1f}\degree$'])
    # cbar.ax.tick_params(labelsize=20)  # Set font size for color bar
    # y axis min max
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Spatial Frequency [cycles/mm]')
    ax.set_ylabel('MTF')
    ax.set_title('MTF vs Frequency')
    
    # return figure for saving 
    return fig

# ==================================
# Point Sampling
# ==================================
@torch.no_grad()
def sample_pts_grid_2D(grid_size=(9,9), align_corner=True, distribution='uniform'):
    """ sample point grid [-1: 1] * [-1: 1] at location z. x from [-1, 1], y from [-1, 1].

    Args:
        depth (float): Depth of the point source plane.
        grid_size (tuple, or Torch Tensor): Size of the grid, can be a tuple (H, W) or a single int for square grid.
        normalized (bool): Whether to use normalized x, y corrdinates [-1, 1]. Defaults to True.
        align_corner (bool): Whether to use center of each patch. Defaults to False.
        distribution (str): Distribution of the point source. Defaults to 'uniform', can be 'sqrt' to focus more on the boundary.

    Returns:
        point_source: Shape of [H, W, 2].
    """
    H,W = grid_size
    # ==> Use corner
    if align_corner:
        x, y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, W), 
            torch.linspace(-1.0, 1.0, H), 
            indexing='xy') # of shape [H, W]
    # ==> Use center of each patch  
    else:   
        # half_bin_size
        hbs_h = 0.5 / H
        hbs_w = 0.5 / W
        x, y = torch.meshgrid(
            torch.linspace(-1 + hbs_w, 1 - hbs_w, W), 
            torch.linspace(-1 + hbs_h, 1 - hbs_w, H), 
            indexing='xy') # of shape [H, W]

    if distribution == 'sqrt':
        x = torch.sqrt(x.abs()) * x.sign()
        y = torch.sqrt(y.abs()) * y.sign()
    
    point_source = torch.stack([x, y], dim=-1)

    return point_source.float()


# Pupil sampling is usually performed in a circular way
@torch.no_grad()
def sample_pts_pupil_1D(spp=16, axis='x', r_range=[0,1],r_space='linear'):
    """ Sample points on one axis, either 'x' or 'y'.

    Args:
        spp (int): sample per pixel. Defaults to 16.
        axis (str): axis to sample, 'x' or 'y'. Defaults to 'x'.
        r_range (list): range of radius to sample, in [0,1] relative to R. Defaults to [0,1].
        r_space (str): radial space to sample, 'sqrt' for sqrt space, 'linear' for linear space. Defaults to 'linear'.
    
    Returns:
        pts (tensor): sampled points on the pupil plane. Shape of [spp,2].
    """
    if axis not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'.")
    # Sample points on a circle grid
    base_theta = 0
    if axis == 'x':
        base_theta = 0
    else:
        base_theta = np.pi / 2
    assert spp % 2 == 0, "spp should be even for 1D sampling."
    return sample_pts_pupil_grid(grid_size=(1,), spp=spp, num_ang=2, r_range=r_range, base_theta=base_theta, random=False, r_space=r_space).squeeze() # shape [spp, 2]
    
@torch.no_grad()
def sample_pts_pupil_2D(spp=16, num_ang=8, random=False, r_range=[0,1]):
    """ Sample points (not rays) on the pupil plane with rings (radial range). Points have shape [spp, 2]. range [-1, 1] in x and y.

        2*pi is devided into [num_ang] sectors.
        Circle is devided into [spp//num_ang] rings.

    Args:
        spp (int): sample per pixel. Defaults to 16.
        num_ang (int): number of sectors. Defaults to 8.
        r_range (list): range of radius to sample, in [0,1] relative to R. Defaults to [0,1].
    
    Returns:
        pts (tensor): sampled points on the pupil plane. Shape of [spp,2].
    """
    return sample_pts_pupil_grid(grid_size=(1,), spp=spp, num_ang=num_ang, random=random, r_range=r_range).squeeze()

@torch.no_grad()
def sample_pts_pupil_grid(grid_size=(512,512), spp=16, num_ang=8, random=True, r_range=[0,1], base_theta = 0, r_space='sqrt'):
    """ Sample points (not rays) on the pupil plane with rings (radial range). Points have shape [spp, (H, W), 2]. range [-1, 1] in x and y.

        2*pi is devided into [num_ang] sectors.
        Circle is devided into [spp//num_ang] rings.

    Args:
        grid_size (tuple, or Torch Tensor): Size of the grid, can be a tuple (H, W) or a single int for square grid.
        spp (int): sample per pixel. Defaults to 16.
        num_ang (int): number of sectors. Defaults to 8.
        r_range (list): range of radius to sample, in [0,1] relative to R. Defaults to [0,1].
        base_theta (float): base angle to start sampling, in radians. Defaults to 0. in range [0, 2*pi).
        r_space (str): radial space to sample, 'sqrt' for sqrt space, 'linear' for linear space. Defaults to 'sqrt'.
    
    Returns:
        pts (tensor): sampled points on the pupil plane. Shape of [spp, (H, W), 2].
    """
    assert r_space in ['sqrt', 'linear'], "r_space should be 'sqrt' or 'linear'."
    r_min, r_max = r_range
    # => Naive implementation with random sampling
    # if large enough spp or spp is not dividable by num_ang, sample uniformly on the pupil
    if spp % num_ang != 0 or spp >= 10000:
        theta = torch.rand((spp, *grid_size)) * 2 * np.pi
        r2 = torch.rand((spp, *grid_size)) 
        if r_space == 'sqrt':
            r = torch.sqrt(r2)
        elif r_space == 'linear':
            r = r2
        r = r * (r_max - r_min) + r_min  # scale radius to [r_min, r_max]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        pts = torch.stack((x,y), -1)

    # => Sample more uniformly when spp is not large
    else:          
        # ==> For each pixel, sample different points on the pupil
        x, y = [], []
        
        angular_step = 2 * np.pi / num_ang
        if random:
            radial_step = num_ang / spp
        else:
            radial_step = num_ang / (spp-num_ang)
        for i in range(num_ang): # sample on angle
            for j in range(spp//num_ang): # sample on radius
                theta = torch.full((1, *grid_size), i * angular_step) + base_theta
                r2 = torch.full((1, *grid_size), j  * radial_step)
                
                if random: # add random perturbation
                    delta_theta = torch.rand((1, *grid_size)) * angular_step # sample delta_theta from [0, angular_step)
                    theta = delta_theta + theta

                    delta_r2 = torch.rand((1, *grid_size)) * radial_step 
                    r2 = delta_r2 + r2  
                    
                if r_space == 'sqrt':
                    r = torch.sqrt(r2)
                elif r_space == 'linear':
                    r = r2
                r = r * (r_max - r_min) + r_min  # scale radius to [r_min, r_max]
                
                x.append(r * torch.cos(theta))
                y.append(r * torch.sin(theta))
        
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        pts = torch.stack((x,y), -1)

    return pts.float()


# ==================================
# Coordinate conversion
# Lens Coordinates (3D): right-handed, x right, y up, z forward into the image plane, metric unit in mm
# Image Coordinates (2D): x right, y up, z forward into the image plane, metric unit in mm
# Relative Coordinates (2D): x right, y down, within range [-1,1], normalized to the image size
# Pixel Coordinates (2D): x right, y down, within range [0,W-1] and [0,H-1]
# ==================================

def cvt_img2rel(pts_img, sensor_size):
    """ Convert sensor coordinates to relative coordinates.
    
    Args:
        pts_img: torch.tensor of shape [N,2] or [2], where N is the number of points, [x,y] in sensor coordinate, the range for x is [-sensor_size[0]/2,sensor_size[0]/2] and for y is [-sensor_size[1]/2,sensor_size[1]/2]
        sensor_size: [H,W] of the sensor in metric unit (mm) or a single int for square sensor size.
    Returns:
        pts_rel: torch.tensor of shape [N,2] or [2], [x,y] in relative coordinate, the range for both width and height is [-1,1], notice y is inverted compared to sensor coordinate.
    """
    if isinstance(sensor_size, int):
        sensor_size = [sensor_size, sensor_size]
    else:
        assert len(sensor_size) == 2 , "sensor_size should be a list of 2 ints or an int"
    sensor_h,sensor_W = sensor_size
    
    assert pts_img.shape[-1] == 2, "pts_img should be of shape [N,2] or [2]"
    
    pts_rel = pts_img / torch.tensor([sensor_W/2,-sensor_h/2]).to(pts_img.device)  # convert to relative image coordinate
    return pts_rel

def cvt_rel2img(pts_rel, sensor_size):
    """ Convert relative coordinates to image coordinates.
    
    Args:
        pts_rel: torch.tensor of shape [N,2] or [2], where N is the number of points, [x,y] in relative image coordinate, the range for both width and height is [-1,1]
        sensor_size: [H,W] of the sensor in metric unit (mm) or a single int for square sensor size.
    Returns:
        pts_img: torch.tensor of shape [N,2] or [2], [x,y] in image coordinate, the range for x is [-sensor_size[0]/2,sensor_size[0]/2] and for y is [-sensor_size[1]/2,sensor_size[1]/2], notice y is inverted compared to image coordinate.
    """
    if isinstance(sensor_size, int):
        sensor_size = [sensor_size, sensor_size]
    else:
        assert len(sensor_size) == 2 , "sensor_size should be a list of 2 ints or an int"
    sensor_h,sensor_W = sensor_size
    
    assert pts_rel.shape[-1] == 2, "pts_rel should be of shape [N,2] or [2]"
    
    pts_img = pts_rel * torch.tensor([sensor_W/ 2, -sensor_h/2]).to(pts_rel.device)  # convert to sensor coordinate
    return pts_img

def cvt_rel2pix(pts_rel, img_res):
    """ Convert relative coordinates to pixel coordinates.
    
    Args:
        pts_rel: torch.tensor of shape [N,2] or [2], where N is the number of points, [x,y] in relative coordinate, the range for both width and height is [-1,1]
        img_res: [H,W] of the image
    Returns:
        pts_pix: torch.tensor of shape [N,2], [u,v] in pixel coordinate, the range for u is [0,W-1] and for v is [0,H-1]
    """
    if isinstance(img_res, int):
        img_res = [img_res, img_res]
    else:
        assert len(img_res) == 2 , "img_res should be a list of 2 ints or an int"
    H,W = img_res
    
    assert pts_rel.shape[-1] == 2, "pts_rel should be of shape [N,2] or [2]"

    pts_pix = (pts_rel+1)/2 * torch.tensor([W,H]).to(pts_rel.device)  # convert to pixel coordinate
    return pts_pix
    

def cvt_pix2rel(pts_pix, img_res):
    """ Convert pixel coordinates to relative coordinates.
    
    Args:
        pts_pix: torch.tensor of shape [N,2] or [2], where N is the number of points, [u,v] in pixel coordinate, the range for u is [0,W-1] and for v is [0,H-1]
        img_res: [H,W] of the image
    Returns:
        pts_rel: torch.tensor of shape [N,2] or [2], [x,y] in relative coordinate, the range for both width and height is [-1,1]
    """
    if isinstance(img_res, int):
        img_res = [img_res, img_res]
    else:
        assert len(img_res) == 2 , "img_res should be a list of 2 ints or an int"
    H,W = img_res
    
    assert pts_pix.shape[-1] == 2, "pts_pix should be of shape [N,2] or [2]"
    
    pts_rel = pts_pix / torch.tensor([W-1,H-1]).to(pts_pix.device) * 2 - 1  # convert to relative image coordinate
    return pts_rel

def cvt_img2pix(pts_img, img_res, sensor_size):
    """ Convert sensor coordinates to pixel coordinates.
    
    Args:
        pts_img: torch.tensor of shape [N,2] or [2], where N is the number of points, [x,y] in sensor coordinate, the range for x is [-sensor_size[0]/2,sensor_size[0]/2] and for y is [-sensor_size[1]/2,sensor_size[1]/2]
        img_res: [H,W] of the image
        sensor_size: [h,w] of the sensor in metric unit (mm) or a single int for square sensor size.
    Returns:
        pts_pix: torch.tensor of shape [N,2] or [2], [u,v] in pixel coordinate, the range for u is [0,W-1] and for v is [0,H-1]
    """
    pts_rel = cvt_img2rel(pts_img, sensor_size)
    pts_pix = cvt_rel2pix(pts_rel, img_res)
    return pts_pix


def cvt_pix2img(pts_pix, img_res, sensor_size):
    """ Convert pixel coordinates to sensor coordinates.
    
    Args:
        pts_pix: torch.tensor of shape [N,2] or [2], where N is the number of points, [u,v] in pixel coordinate, the range for u is [0,W-1] and for v is [0,H-1]
        img_res: [H,W] of the image
        sensor_size: [h,w] of the sensor in metric unit (mm) or a single int for square sensor size.
    Returns:
        pts_img: torch.tensor of shape [N,2] or [2], [x,y] in sensor coordinate, the range for x is [-sensor_size[0]/2,sensor_size[0]/2] and for y is [-sensor_size[1]/2,sensor_size[1]/2]
    """
    pts_rel = cvt_pix2rel(pts_pix, img_res)
    pts_img = cvt_rel2img(pts_rel, sensor_size)
    return pts_img

# ==================================
# roi related
# ==================================
def get_roi_center(roi):
    """ Get the center of the roi.
    
    Args:
        roi (list): [x, y, w, h], where (x,y) is the top-left corner of the roi, w is the width, and h is the height.
    Returns:
        center (tuple): (x, y) coordinates of the center of the roi.
    """
    x, y, w, h = roi
    center_x = x + w / 2
    center_y = y + h / 2
    return (center_x, center_y)

def roi_rel2pix(roi_rel,img_res):
    """ Get the roi from relative coordinate to image pixel coordinate
        for example for img_res=[2048,2048] image, roi_rel=[-1,-1,2,2] will return [0,0,2048,2048]

    Args:
        roi_rel: [N,4] or [4] in relative coordinate of (x,y,w,h), the range for both width and height is [-1,1]
        img_res: [H,W] of the image
    Returns:
        roi_pix: [N,4] or [4] in pixel coordinate of (x,y,w,h)
    """
    if isinstance(img_res, int):
        img_res = [img_res, img_res]
    else:
        assert len(img_res) == 2 , "img_res should be a list of 2 elements"
    H,W = img_res

    if type(roi_rel) is not torch.Tensor:
        roi_rel = torch.tensor(roi_rel)
    assert roi_rel.shape[-1] == 4, "roi_rel should be of shape [N,4] or [4]"
    
    # crop only the interested roi
    x,y,w,h = roi_rel.unbind(dim=-1)
    min_x,max_x = (x+1)/2 *W , (x+w+1)/2 * W
    min_y,max_y = (y+1)/2 *H , (y+h+1)/2 * H
    
    roi_pix = torch.stack([min_x, min_y, max_x-min_x, max_y-min_y], dim=-1).int()
    return roi_pix

def roi_pix2rel(roi_pix,img_res):
    """ Get the roi from pixel coordinate to relative coordinate
        for example for img_res=[2048,2048] image, roi_pix=[0,0,2048,2048] will return [-1,-1,2,2]

    Args:
        roi_pix: [N,4] or [4] in pixel coordinate of (x,y,w,h)
        img_res: [H,W] of the image
    Returns:
        roi_rel: [N,4] or [4] in relative coordinate of (x,y,w,h), the range for both width and height is [-1,1]
    """
    if isinstance(img_res, int):
        img_res = [img_res, img_res]
    else:
        assert len(img_res) == 2 , "img_res should be a list of 2 elements"
    H,W = img_res

    assert roi_pix.shape[-1] == 4, "roi_pix should be of shape [N,4] or [4]"
    
    # crop only the interested roi
    x,y,w,h = roi_pix.unbind(dim=-1)
    min_x,max_x = x / W * 2 - 1 , (x+w) / W * 2 - 1
    min_y,max_y = y / H * 2 - 1 , (y+h) / H * 2 - 1
    
    roi_rel = torch.stack([min_x, min_y, max_x-min_x, max_y-min_y], dim=-1)
    return roi_rel

def random_roi(w=0.5, h=0.5):
    '''
        generate a random relative roi region
        x in [-1,1-w], y in [-1,1-h]
    '''
    x = random.uniform(-1,1-w)
    y = random.uniform(-1,1-h)
    return [x,y,w,h]


# misc
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    # Example usage
    # pts_pix= torch.tensor([[0, 0], [1024, 1024], [2048, 0]])
    pts_pix = torch.tensor([0, 0])
    img_res = [2048,2048]
    sensor_size = [11.264, 11.264]  # Example sensor size in mm
    pts_rel = cvt_pix2rel(pts_pix, img_res)
    print("Converted to image coordinates:\n", pts_rel)

    pts_img = cvt_rel2img(pts_rel, sensor_size)
    print("Converted to sensor coordinates:\n", pts_img)
    
    pts_rel_converted = cvt_img2rel(pts_img, sensor_size)
    print("Converted back to relative coordinates:\n", pts_rel_converted)
    
    ptr_pix_converted = cvt_rel2pix(pts_rel, img_res)
    print("Converted back to pixel coordinates:\n", ptr_pix_converted)
    
    # example for roi
    roi_rel = torch.tensor([-0.5, -0.5, 1.0, 1.0])  # Example relative roi
    roi_pix = roi_rel2pix(roi_rel, img_res)
    print("Converted roi to pixel coordinates:\n", roi_pix)
    
    roi_rel_converted = roi_pix2rel(roi_pix, img_res)
    print("Converted roi back to relative coordinates:\n", roi_rel_converted)