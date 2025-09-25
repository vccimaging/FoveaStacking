import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Sharpness Loss (Laplacian related)
# ======================================================

def gaussian_kernel(size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    interval  = (2*sigma +1)/(size)
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    kernel_tensor.to(device)
    return kernel_tensor

def gaussian_conv2d(x, g_kernel, dtype=torch.float):
    #Assumes input of x is of shape: (minibatch, depth, height, width)
    #Infer depth automatically based on the shape
    channels = g_kernel.shape[0]
    padding = g_kernel.shape[-1] // 2 # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    x_pad = F.pad(x, (padding, padding, padding, padding), mode='replicate') # Pad the image with replicate mode
    y = F.conv2d(x_pad, weight=g_kernel, stride=1, padding="valid", groups=channels)
    return y

def downsample(x):
    # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    return x[:, :, ::2, ::2]

def create_laplacian_pyramid(x, kernel, levels):
    upsample = torch.nn.Upsample(scale_factor=2) # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    H,W = x.shape[-2:]
    H_pad = 2**levels - H % 2**levels
    W_pad = 2**levels - W % 2**levels
    x_pad = F.pad(x, (0, W_pad, 0, H_pad), mode='replicate') # Pad the image with replicate mode
    current_x = x_pad
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        # Original Algorithm does indeed: L_i  = G_i  - expand(G_i+1), with L_i as current laplacian layer, and G_i as current gaussian filtered image, and G_i+1 the next.
        # Some implementations skip expand(G_i+1) and use gaussian_conv(G_i). We decided to use expand, as is the original algorithm
        laplacian = current_x - upsample(down) 
        pyramids.append(laplacian)
        current_x = down 
    pyramids.append(current_x)
    return pyramids

class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, kernel_size=3, sigma=1, dtype=torch.float,device=torch.device('cpu')):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype).to(device)
    
    def forward(self, x, batch_mean=True, verbose=False):
        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels) # a list of max_levels of image pyramids (B,C, H_level, W_level)
        mean_L2 = []
        for i in range(0, self.max_levels):
            mean_L2.append((input_pyramid[i]**2).mean(axis=(1,2,3)) * 2**(self.max_levels - i)) # each level is weighted by 2^(max_levels - i), has shape (B,)
        mean_L2 = torch.stack(mean_L2).mean(axis=0) #  stack to have shape (max_levels, B), then mean to have shape (B,)
        if batch_mean:
            mean_L2 = mean_L2.mean()
        if verbose:
            return mean_L2, input_pyramid
        
        loss = -mean_L2 # Maximizing the mean L2 of the Laplacian pyramid
        return loss 
    
    
def LaplacianVarLoss(img):
    """
    Computes the variance of the Laplacian of the image, should encouraging higher variance (sharper image).
    """
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        laplacian_kernel = laplacian_kernel.cuda()
    laplacian_img = F.conv2d(img, laplacian_kernel, padding=1, groups=img.shape[1])
    variance = torch.var(laplacian_img)
    loss = -variance  # Maximizing variance
    return loss

def total_variation_loss(img, beta=2):
    """
    Computes the Total Variation Loss for an image, which can be used to encourage sharpness.
    """
    dh = torch.pow(img[:, :, :-1] - img[:, :, 1:], 2)
    dw = torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2)
    loss = -torch.sum(torch.pow(dh + dw, beta / 2.0))
    return loss


# ======================================================
# Edge aware Loss (using sobel)
# ======================================================
class TenengradLoss(torch.nn.Module):
    def __init__(self):
        super(TenengradLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_x, self.sobel_y = self.sobel_x.repeat(3, 1, 1, 1), self.sobel_y.repeat(3, 1, 1, 1)
    
    def forward(self, img):
        """
        Computes the Tenengrad variance loss, another measure of sharpness based on the gradient. Should encourage higher value of it.
        """
        assert len(img.size()) == 4 # (batch, channels, height, width)
        assert img.size(1) == 3 # RGB image
        if img.is_cuda:
            self.sobel_x, self.sobel_y = self.sobel_x.cuda(), self.sobel_y.cuda()
        grad_x = F.conv2d(img, self.sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, self.sobel_y, padding=1, groups=img.shape[1])
        tenengrad = grad_x**2 + grad_y**2
        loss = -torch.mean(tenengrad)  # Maximizing tenengrad
        return loss


def edge_detection_loss(img):
    """
    Computes loss based on edge detection (Sobel filter).
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()
    edge_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    edge_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    loss = -torch.mean(edges)  # Maximizing edge response
    return loss



# ======================================================
# PSNR and SSIM loss
# ======================================================
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target, batch_mean=True):
        return 1 - ssim(pred, target, size_average= batch_mean)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) # (B, C, H_out, W_out)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) # (B, C, H_out, W_out)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2) # (B, C, H_out, W_out)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret 


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


