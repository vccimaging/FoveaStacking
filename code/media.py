import ffmpeg
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil
import os
import subprocess

def imgs2video(imgs, gif_name='vis.gif', frame_rate=3):
    ''' Create gif from images
    Args:
        imgs (str): path to images, e.g. './results/psf/test_*mm.png'
        gif_name (str): name of gif file, e.g. 'vis.gif'
        
    '''
    # create gif from images
    (
    ffmpeg
    .input(imgs, pattern_type='glob', framerate=frame_rate)
    .output(gif_name)
    .run()
    )
    

def resize_img(img, align="horizontal", res=2048):
    ''' Resize image
    Args:
        img (PIL.Image): image
        align (str): 'horizontal' or 'vertical'
    '''
    # resize images
    width, height = img.size
    if align=='horizontal':
        img = img.resize((int(width*res/height), res))
    else:
        img = img.resize((res, int(height*res/width)))
    return img

def combine_imgs(imgs_list, imgs_dst, align='horizontal',res=512):
    ''' Combine images
    Args:
        imgs_list (str): list of path to images, e.g. ['./results/psf/test_u*mm.png',]
        imgs_dst (str): path to images, e.g. './results/psf/test_*mm.png'
        align (str): 'horizontal' or 'vertical'
        
    '''
    # combine images
    fname_list =[]
    for imgs in imgs_list:
        fname = sorted(glob.glob(imgs))
        print(f"loading fname: {fname}")
        fname_list.append(fname)

    n_type = len(fname_list)    # number of different typesimages
    n_frame = len(fname_list[0]) # number of frames
    res_combine=0
    for i in range(n_type):
        assert len(fname_list[i])==n_frame, f"Number of frames in {i}th type is not equal to {n_frame}, but {len(fname_list[i])}"
        img = Image.open(fname_list[i][0])
        width,height = img.size
        if align=='horizontal':
            res_combine+=int(width*res/height)
        else:
            res_combine+=int(height*res/width)


    # initialize new image
    if res_combine%2!=0:
        res_combine+=1
    if align=='horizontal':
        new_img = Image.new('RGB', (res_combine, res),color=(255,255,255))
    else:
        new_img = Image.new('RGB', (res, res_combine),color=(255,255,255))

    
    for k in range(n_frame):
        acc_res = 0
        for f in range(n_type):
            img = resize_img(Image.open(fname_list[f][k]), align=align, res=res)
            if align=='horizontal':
                new_img.paste(img, (acc_res, 0))
                acc_res+=img.size[0]
            else:
                new_img.paste(img, (0, acc_res))
                acc_res+=img.size[1]
            
        new_img.save(imgs_dst.replace('*', f"{k:03d}"))

if __name__=="__main__":
    result_dir = 'results/0924-115201-opt_DPP_joint_5DPPs' # path to result directory
    n_dpps = 5
    
    img_list = [f"{result_dir}/psf_val/attention_map_*.png",
                f"{result_dir}/psf_val/min_RMS_*.png",
                ]
    for i in range(1,n_dpps+1):
        img_list.append(f"{result_dir}/psf_val/RMS_lens0{i}_*.png")
    combine_imgs(img_list,f'{result_dir}/combine_*.png', align='horizontal')    
    imgs2video(f'{result_dir}/combine_*.png', f'{result_dir}/combine.mp4', frame_rate=2)