""" 
Jointly optimize DPP patterns to cover the entire image Field of View (FoV) using differentiable rendering.

Support multiple GPUs for increasing number of DPPs.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # specify which GPU(s) to use

import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse
from deeplens import *    
import shutil
from imageio import imread
from media import combine_imgs,imgs2video
import matplotlib as mpl
from matplotlib.colors import  ListedColormap

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
    
def save_lens_list(lens_list, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i,lens in enumerate(lens_list):
        lens.write_lens_json(f'{result_dir}/lens_{i+1}.json')

def validate(lens_list, 
             result_dir, 
             res=[1024,1024], 
             psf_stat='rms', 
             render=False, 
             analyze=False,
             n_psf_grid = 32,
             iter=0):
    with torch.no_grad():
        val_img_res = res

        os.makedirs(f"{result_dir}/lenses",exist_ok=True)
        os.makedirs(f"{result_dir}/psf_val",exist_ok=True)

        metrics_dict = {}
        for i in range(len(lens_list)):
            metrics_dict[f'psf_{psf_stat}_{i}'] = []

        
        if render:
            for i in range(len(lens_list)):
                metrics_dict[f'psnr_{i}'] = []
                metrics_dict[f'ssim_{i}'] = []
            os.makedirs(f"{result_dir}/full_render",exist_ok=True)
            os.makedirs(f"{result_dir}/masked_render",exist_ok=True)

            # quantiative evaluation
            val_dataset = SingleImageDataset('data/single_image/new_york.jpg')
            data = val_dataset[0]
            img_ref = data['img_ref'].unsqueeze(0) # .to(device)

            # => Render pinhole image
            img_pinhole = lens_list[0].render(img_ref.to(lens_list[0].device), method='pinhole', sensor_res=val_img_res)
            # img_pinhole = img_pinhole**(1/2.2) # gamma correction
            img_pinhole = lin2srgb(img_pinhole) # convert to sRGB
            save_image(img_pinhole, f'{result_dir}/full_pinhole.png')

        psf_list = []
        for lens_id,lens in enumerate(lens_list):
            if render:
                # ==> render and compare the image
                img_render = lens.render(img_ref.to(lens.device), sensor_res=val_img_res)
                # img_render = img_render**(1/2.2) # gamma correction
                img_render = lin2srgb(img_render) # convert to sRGB
                save_image(img_render, f'{result_dir}/full_render/lens{lens_id:02d}_iter{iter}.png')
                
                psnr_render = batch_PSNR(img_pinhole.to(lens.device), img_render)
                ssim_render = batch_SSIM(img_pinhole.to(lens.device), img_render).item()
                metrics_dict[f'psnr_{lens_id}'].append(psnr_render)
                metrics_dict[f'ssim_{lens_id}'].append(ssim_render)

            #  ==> evaluate the psf
            psfs, pointc_sensor,point_stats,valid_index = lens.psf_map(grid=n_psf_grid, ks=5, spp=GEO_SPP, stats_only=True)
            psf_val = point_stats[psf_stat].cpu().mean(axis=-1) # gather info in cpu, psf stats has mean, geo_mean, mse
            psf_list.append(psf_val) # average over the RGB channels
            metrics_dict[f'psf_{psf_stat}_{lens_id}'].append(psf_val.mean().item())
            
            # 2D draw the psf
            psf_grid = psf_val.reshape(n_psf_grid,n_psf_grid)
            draw_val_2D(psf_grid*1000, # in um
                        title=f'Lens{lens_id+1:02d} RMS (um)',
                        vmin=0,
                        vmax=50,
                        fig_name=f'{result_dir}/psf_val/RMS_lens{lens_id+1:02d}_iter{iter}.png',
                        cmap='hot')

            # => Save data and simple evaluation
            lens.write_lens_json(f'{result_dir}/lenses/lens_{lens_id+1:02d}_iter{iter}.json')
            if analyze:
                os.makedirs(f"{result_dir}/lenses_analysis",exist_ok=True)
                lens.analysis(f'{result_dir}/lenses_analysis/lens_{lens_id+1:02d}_iter{iter}')
            
        # save the psf_list
        psf_stack = torch.stack(psf_list) # shape (n_lens,n_psf_grid,n_psf_grid)
        np.save(f'{result_dir}/psf_stack_iter{iter}.npy',psf_stack.numpy())

        # ==> Post analysis:
        # visualize the attention map
        min_val, min_ind = torch.min(psf_stack,dim=0) # average over the stack
        metrics_dict['PSF_min'] = [(min_val.mean().cpu()).item()]

        attention_map = min_ind.detach().reshape(n_psf_grid,n_psf_grid) # shape (n_psf_grid,n_psf_grid)
        plot_class(f'{result_dir}/psf_val/attention_map_iter{iter}.png',
                   attention_map.cpu().numpy(), # add 1 to make the class start from 1
                   title='Attention Map',
                   n_class=len(lens_list))

        # visualize the min psf value
        psf_grid = min_val.cpu().reshape(n_psf_grid,n_psf_grid)
        draw_val_2D(psf_grid*1000, # in um
                    title=f'Min RMS Spot Size (um)',
                    vmin=0,
                    vmax=50,
                    fig_name=f'{result_dir}/psf_val/min_RMS_iter{iter}.png',
                    cmap='hot')
        
        # Visualization the stacking
        attention_map_full = nn.functional.interpolate(attention_map.unsqueeze(0).unsqueeze(0).float(),size=val_img_res,mode='nearest').squeeze(0).squeeze(0).long().cpu() # shape (H,W)
        np.save(f'{result_dir}/attention_map',attention_map_full.cpu().numpy())
        
        if render:
            fuse_img = torch.zeros([3,val_img_res[0],val_img_res[1]])
            for lens_id,lens in enumerate(lens_list):
                img_render = imread(f'{result_dir}/full_render/lens{lens_id:02d}_iter{iter}.png') # shape (H,W,C)
                img_render = torch.tensor(img_render).permute(2,0,1).float()/255 # shape (C,H,W)
                fuse_img[:,attention_map_full==lens_id] = img_render[:,attention_map_full==lens_id]
                img_render.masked_fill_(attention_map_full!=lens_id,0)
                save_image(img_render, f'{result_dir}/masked_render/lens{lens_id:02d}_iter{iter}.png')
            save_image(fuse_img, f'{result_dir}/fuse_render_iter{iter}.png')
            metrics_dict['psnr_fuse'] = [batch_PSNR(img_pinhole.cpu(), fuse_img.unsqueeze(0))]
            metrics_dict['ssim_fuse'] = [batch_SSIM(img_pinhole.cpu(), fuse_img.unsqueeze(0)).item()]
            
        for key in metrics_dict:
            metrics_dict[key] = np.mean(metrics_dict[key])

        info_str = ', '.join([f'{key}: {metrics_dict[key]:.4f}' for key in metrics_dict])
        logging.info(f'Validating: {info_str}')
        
        # Save metrics to json, if the json file does not exist, create a new one
        if not os.path.exists(f'{result_dir}/validation.json'):
            with open(f'{result_dir}/validation.json', 'w') as f:
                json.dump({f"val_iter{iter}":metrics_dict}, f, indent=2)
        else:
            with open(f'{result_dir}/validation.json', 'r') as f:
                data = json.load(f)
            with open(f'{result_dir}/validation.json', 'w') as f:
                validate_json = data
                validate_json.update({f"val_iter{iter}":metrics_dict})  
                json.dump(validate_json, f, indent=2)

        return metrics_dict


def opt_dpp_joint(lens_list, args, result_dir = None, psf_stat='rms'):
    if result_dir is None:
        result_dir = args['result_dir']

    # ==> Lens optimizer
    lens_params = {}
    for i,lens in enumerate(lens_list):
        device = lens.device
        if device not in lens_params: # create a new list for each device
            lens_params[device] = []
        if i>=0: # skip (or not) the first lens
            lens_params[device] += lens.get_parameters(lr=args['lens']['lr'])
    
    # ==> Network optimizer
    n_epochs = args['train']['epochs']
    lens_optim = {}
    for device,params in lens_params.items():
        lens_optim[device] = torch.optim.Adam(params) 

    # ==> Log
    logging.info(f'Start Joint Optimization.')
    
    # ==> Criterion
    step = 0
    loss_best = 1e10

    # ==> Training
    for epoch in range(n_epochs+1):
        if epoch % 20 == 0:
            validate(lens_list, result_dir, res=[1024,1024], render=False,psf_stat=psf_stat,iter=epoch)

        torch.autograd.set_detect_anomaly(True) 
        
        # ==> Train 1 epoch
        Loss_dict = {}

        # PSF RMS loss, require known depth
        n_grid = args['n_grid']
        psf_list = []
        roi_rel = [-1,-1,2,2]        
        for i,lens in enumerate(lens_list):
            psfs, pointc_sensor,point_stats,valid_index = lens.psf_map(grid=n_grid, roi_rel=roi_rel, spp=int(GEO_SPP/4), stats_only=True,distribution='uniform') # distribution can be "sqrt" or "uniform"
            psf_stats = point_stats[psf_stat].mean(axis=-1) # average across RGB wavelength
            psf_list.append((psf_stats).cpu())  # gather info in cpu, psf stats has mean, geo_mean, mse

        psf_grids = torch.stack(psf_list).reshape(-1,n_grid,n_grid) # shape (n_lens,n_grid,n_grid)
        # create mask from valid_index
        valid_mask = torch.zeros(n_grid,n_grid)
        valid_mask.flatten()[valid_index.cpu()] = 1
        valid_mask = valid_mask.bool() # shape (n_grid,n_grid)
        
        min_val, min_ind = torch.min(psf_grids,dim=0) # average over the stack, each of shape (n_grid, n_grid)
        Loss_dict['PSF'] = min_val[valid_mask].mean()
        
        # enforce the unused lens to optimize on the difficult region
        for i,lens in enumerate(lens_list):
            if (min_ind[valid_mask]==i).sum() == 0: # if the lens is not used
                # min_val/min_val.mean() here acts like a weight\
                weight = ((min_val)/min_val.mean()).detach()
                Loss_dict[f'PSF_{i}'] = (psf_grids[i] * weight).mean() 
        
        Loss = 0
        for key in Loss_dict:
            Loss += Loss_dict[key]

        # Save best model
        if Loss < loss_best:
            loss_best = Loss
            save_lens_list(lens_list, f'{result_dir}/best')

        # => Back-propagation
        # ========================================
        # Line 4: zero-grad
        # ========================================
        for device,optim in lens_optim.items():
            optim.zero_grad()

        Loss.backward()
        step += 1

        # ========================================
        # Line 5: step
        # ========================================
        for device,optim in lens_optim.items():
            optim.step()
            for lens in lens_list:
                if device == lens.device:
                    lens.update()
                    
        loss_str = ', '.join([f'{key}: {Loss_dict[key].detach().item():.4f}' for key in Loss_dict])
        logging.info(f'Epoch [{epoch+1}/{args["train"]["epochs"]}]] loss:{Loss.detach().item():.4f}, {loss_str}')

if __name__=='__main__':
    # ==> Parse arguments
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, default='configs/optimize_60m.yml')
    argparse.add_argument('--exp_name', type=str, default='opt_DPP_joint')
    argparse.add_argument('--n_dpps', type=int, default=5, help='number of DPPs to jointly optimize')
    argparse.add_argument('--psf_stat', type=str, default='rms', help='use RMS spot size to optimize the lens')
    argparse.add_argument('--defocus_only', action='store_true', help='if true, only optimize the defocus term of the DPP')
    argparse.add_argument('--load_lens', type=str, default=None,help='load external lens path that can override the config file as initial settings')
    opt = argparse.parse_args()

    # # ==> Config
    opt.exp_name+=f"_{opt.n_dpps}DPPs"
    args = config(file_path=opt.config,EXP_NAME=opt.exp_name)
    
    # save args to a new config file
    with open(os.path.join(args['result_dir'], 'configs','config.yml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)
    shutil.copy(__file__, os.path.join(args['result_dir'],os.path.basename(__file__))) # save code

    # ==> get GPU info
    n_gpus = torch.cuda.device_count()
    logging.info(f"Using {n_gpus} GPUs for model parallel training")

    # ==> initialize all the lenses 
    lens_list = []
    for i in range(1,opt.n_dpps+1):
        if n_gpus > 0:
            gpu_id = i % n_gpus
            if opt.load_lens is not None:
                lens = Lensgroup(filename=f'{opt.load_lens}/lens_{i}.json', sensor_res=args['img_res'],device=f'cuda:{gpu_id}')
            else:
                lens = Lensgroup(filename=args['lens']['path'], sensor_res=args['img_res'],device=f'cuda:{gpu_id}')
        else:
            if opt.load_lens is not None:
                lens = Lensgroup(filename=f'{opt.load_lens}/lens_{i}.json', sensor_res=args['img_res'],device='cpu')
            else:
                lens = Lensgroup(filename=args['lens']['path'], sensor_res=args['img_res'],device='cpu')
            
            # Add random noise to the lens zernike parameters
            zern_amp = lens.surfaces[lens.dpp_idx].get_zern_amp()
            zern_amp = zern_amp + torch.randn_like(zern_amp)*1e-5
            lens.surfaces[lens.dpp_idx].set_zern_amp(zern_amp)

        if opt.defocus_only:    # update the lens to defocus only
            lens.surfaces[lens.dpp_idx].defocus_only = True
        
        lens_list.append(lens)
    
    # ==> run experiment
    opt_dpp_joint(lens_list, args, psf_stat=opt.psf_stat)

    # ==> validate the best model
    best_result_dir = os.path.join(args['result_dir'], 'best')
    
    # load the best model
    lens_list = []
    for i in range(1,opt.n_dpps+1):
        if n_gpus > 0:
            gpu_id = i % n_gpus
            lens = Lensgroup(filename=f'{best_result_dir}/lens_{i}.json', sensor_res=args['img_res'],device=f'cuda:{gpu_id}')
        else:
            lens = Lensgroup(filename=f'{best_result_dir}/lens_{i}.json', sensor_res=args['img_res'],device='cpu')

        lens_list.append(lens)

    # validate the best model
    metrics_dict = validate(lens_list, best_result_dir, res=args['img_res'], render=True, analyze=True, psf_stat=opt.psf_stat)
    
    # combine the images to generate a video
    result_dir = args['result_dir']
    img_list = [f"{result_dir}/psf_val/attention_map_*.png",
                f"{result_dir}/psf_val/min_RMS_*.png",
                ]
    for i in range(1,opt.n_dpps+1):
        img_list.append(f"{result_dir}/psf_val/RMS_lens0{i}_*.png")
    combine_imgs(img_list,f'{result_dir}/combine_*.png', align='horizontal')    
    imgs2video(f'{result_dir}/combine_*.png', f'{result_dir}/combine.mp4', frame_rate=2)