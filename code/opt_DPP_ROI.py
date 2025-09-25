""" 
optimize a single DPP pattern to correct local aberrations within a Region of Interest (ROI) using differentiable rendering.
"""

import os
import logging
import torch
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
from deeplens import *    
import shutil

def opt_dpp(lens, args,result_dir=None, roi_rel=[-1,-1,0.5,0.5], psf_stat='mean'):
    if result_dir is None:
        result_dir = args['result_dir']

    # ==> Lens optimizer
    lens_params = lens.get_parameters(lr=args['lens']['lr'])
    lens_optim = torch.optim.Adam(lens_params)

    # ==> Log
    logging.info(f'Start optical design.')
    
    # ==> Criterion
    step = 0
    loss_best = 1e10
    lens.write_lens_json(f'{result_dir}/best.json')
    # ==> Training
    for epoch in range(args['train']['epochs_roi']):
        # ==> Train 1 epoch
        torch.autograd.set_detect_anomaly(True) 
        Loss_dict = {}

        # PSF MSE loss, require known depth
        psfs_valid, pointc_sensor,points_stats,_ = lens.psf_map(grid=args['n_grid_roi'], ks=3, spp=GEO_SPP,roi_rel=roi_rel, stats_only=True)
        Loss_dict['PSF_radius'] = torch.mean(points_stats[psf_stat])
        
        Loss = 0
        for key in Loss_dict:
            Loss += Loss_dict[key]

        # Save best model
        if Loss < loss_best:
            loss_best = Loss
            lens.write_lens_json(f'{result_dir}/best.json')
        lens_optim.zero_grad()

        Loss.backward()
        step += 1

        # ========================================
        # Line 5: step
        # ========================================
        lens_optim.step()
        lens.update()

        # logging.info({"loss_class":Loss.detach().item()})
        loss_str = ', '.join([f'{key}: {Loss_dict[key].detach().item():.4f}' for key in Loss_dict])
        logging.info(f'Epoch [{epoch+1}/{args["train"]["epochs_roi"]}], loss:{Loss.detach().item():.4f}, {loss_str}')

def render_img(val_img_res, roi_rel, file_name, device):
    # ==> Dataset
    data_set = SingleImageDataset('data/single_image/new_york.jpg')
    data_loader = DataLoader(data_set, batch_size=1)

    with torch.no_grad():
        # roi_rel to roi_pix
        roi_pix = roi_rel2pix(roi_rel, val_img_res)
        x,y,w,h = roi_pix.tolist()

        # get first data
        data = next(iter(data_loader))
        img_ref= data['img_ref'].to(device)

        # => Render image
        img_render = lens.render(img_ref, spp=128, roi_pix=None, sensor_res=val_img_res)
        img_render = lin2srgb(img_render) # convert to sRGB
        file_dir,file_name = os.path.split(file_name)
        save_image(img_render[...,y:y+h,x:x+w], f'{file_dir}/render_crop_{file_name}.png')
        save_image(img_render, f'{file_dir}/render_full_{file_name}.png')
        
        # add red box for ROI
        img_render = img_render[0].cpu().numpy().transpose(1,2,0)
        img_render = (img_render*255).astype(np.uint8)
        img_render = cv.rectangle(img_render.copy(), (x,y), (x+w,y+h), color=(255,0,0), thickness=5)
        cv.imwrite(f'{file_dir}/render_full_{file_name}_box.png', img_render[...,::-1])

def evaluate(lens, result_dir, opt,name='best'):
    lens.analysis(f'{result_dir}/{name}', render=False, roi_rel=opt.ROI)
    lens.draw_spot_diagram(kernel_size=19, spp=2048, grid=5, file_name=f"{result_dir}/{name}_outer_spot_diagram",axis="diag")
    render_img(lens.sensor_res, opt.ROI, f'{result_dir}/{name}', lens.device)
    

if __name__=='__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, default='configs/optimize_652mm.yml')
    argparse.add_argument('--ROI', nargs='+', default=[0.6,0.6,0.4,0.4],help='[x_center, y_center, width, height] in relative sensor coord (range in [-1,1]). If empty, use the default ROI from config')
    argparse.add_argument('--exp_name', type=str, default='opt_DPP_ROI')
    argparse.add_argument('--defocus_only', action='store_true')
    argparse.add_argument('--psf_stat', type=str, default='rms')
    opt = argparse.parse_args()
    opt.exp_name = f'{opt.exp_name}_[{opt.ROI[0]},{opt.ROI[1]},{opt.ROI[2]},{opt.ROI[3]}]'
    opt.ROI = [float(i) for i in opt.ROI]

    args = config(file_path=opt.config,EXP_NAME=opt.exp_name)
    shutil.copy(__file__, os.path.join(args['result_dir'],os.path.basename(__file__))) # save code

    if len(opt.ROI) == 0:
        # use default ROI from config
        opt.ROI = args['ROI']
    elif len(opt.ROI) != 4:
        raise ValueError("ROI should have 4 elements.")
    print(f"ROI: {opt.ROI}")


    # load the lens
    lens = Lensgroup(filename=args['lens']['path'],sensor_res=[1024,1024], device=args['device'])
    evaluate(lens, args['result_dir'], opt,name='init')
    
    if opt.defocus_only:    # update the lens to defocus only
        lens.surfaces[lens.dpp_idx].defocus_only = True

    # run experiment
    opt_dpp(lens, args, roi_rel=opt.ROI,psf_stat=opt.psf_stat)

    # evaluate the best lens with customized visualization 
    lens = Lensgroup(filename=f'{args["result_dir"]}/best.json',sensor_res=[1024,1024], device=args['device'])
    evaluate(lens, args['result_dir'], opt,name='best')
    
    n_grid = 21
    psfs_valid, pointc_sensor,points_stats,_ = lens.psf_map(grid=n_grid, stats_only=False)
    psf_val = torch.mean(points_stats["rms"],dim=-1)*1000 # convert to um
    psf_val = psf_val.detach().cpu().reshape(n_grid,n_grid).numpy()
    draw_val_2D(psf_val,title=f'PSF RMS: {psf_val.mean():.2f} um, max: {psf_val.max():.2f} um',fig_name=f'{args["result_dir"]}/val.png',cmap='hot')
    