""" 
This file contains basic dataset class, used in the AutoLens project.
"""
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
from .utils_io import *

# ======================================
# Basic dataset classes
# ======================================

class SingleImageDataset(Dataset):
    """Single image dataset, load reference and captured image
        Mainly used for single image calibration
    """
    def __init__(self, ref_fname, num_samples=1, cap_fname=None, cap_res = [2048,2048], **kwargs):
        
        # define operations
        self.to_tensor = transforms.ToTensor()
        self.set_resize(cap_res)
        
        self.img_ref = Image.open(ref_fname).convert('RGB')
        self.img_ref = self.to_tensor(self.img_ref)
        self.img_ref = srgb2lin(self.img_ref) # convert to linear RGB

        if cap_fname is not None:
            self.img_cap = Image.open(cap_fname).convert('RGB')
            self.img_cap = self.to_tensor(self.img_cap)
            self.img_cap = srgb2lin(self.img_cap) # convert to linear RGB
        else:
            self.img_cap = self.img_ref
            
        self.num_samples = num_samples
        
        
    def set_resize(self, img_res=[512,512]):
        self.resize = transforms.Resize(img_res, antialias=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_cap = self.resize(self.img_cap)
        return {
            "img_ref":self.img_ref, 
            "img_cap":img_cap
            }

class CalibDataset(Dataset):
    """ load data with different DPP zernike parameters, this requires structured dataset.
        Mainly used for DPP calibration
    """
    def __init__(self, img_dir, img_res=[2048,2048], noise = 0.01, verbose=False, **kwargs):
        super(CalibDataset, self).__init__()
        self.noise = noise
        
        # set transformations
        self.to_tensor = transforms.ToTensor()
        self.set_resize(img_res)

        # read related parameters and black/reference image
        self.img_ref = Image.open(f"{img_dir}/ref.png").convert('RGB')
        self.img_ref = self.to_tensor(self.img_ref)
        self.img_ref = srgb2lin(self.img_ref) # convert to linear RGB

        # read all images and json files
        self.cap_paths = sorted(glob.glob(f"{img_dir}/cap_*.png"))
        self.json_paths = sorted(glob.glob(f"{img_dir}/*.json"))
        assert len(self.cap_paths) == len(self.json_paths), "Number of images and json files should be the same."


    def set_resize(self, img_res=[512,512]):
        self.resize = transforms.Resize(img_res, antialias=True)
    
    def __len__(self):
        return  len(self.cap_paths)

    def __getitem__(self, idx):
        img_cap = Image.open(self.cap_paths[idx]).convert("RGB") # read raw uint16 image
        img_cap = self.to_tensor(img_cap) # to tensor
        img_cap = srgb2lin(img_cap) # convert to linear RGB

        img_cap += self.noise * torch.randn_like(img_cap)
        img_cap = self.resize(img_cap)

        return_dict = {
            "img_ref":self.img_ref, 
            "img_cap":img_cap
            }

        with open(self.json_paths[idx], 'r') as f:
            data = json.load(f)
            return_dict.update(data)

        return return_dict

