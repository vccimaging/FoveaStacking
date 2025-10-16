""" Lensgroup class. Use geometric ray tracing to optical computation.
Code adapted from DeepLens: https://github.com/singer-yang/DeepLens
"""
import torch
import json
import cv2 as cv
import matplotlib.pyplot as plt
import torch.nn.functional as F

from .surfaces import *
from .utils import *
from .monte_carlo import forward_integral
from .basics import GEO_SPP, EPSILON, WAVE_SPEC
from typing import Union, List
class Lensgroup(DeepObj):
    def __init__(self, filename=None, sensor_res=[1024, 1024], device=DEVICE):
        """ Initialize Lensgroup.

        Args:
            filename (string): lens file.
            device ('cpu' or 'cuda'): device.
            sensor_res: (H, W)
        """
        super(Lensgroup, self).__init__()
        
        # Load lens file.
        if filename is not None:
            self.lens_name = filename
            self.device = device
            self.load_file(filename, device=device)
            self.to(device)

            # prepare sensor 
            self.prepare_sensor(sensor_res, sensor_size=self.sensor_size)    
        
        else:
            self.sensor_res = sensor_res
            self.surfaces = []
            self.materials = []

    def set_res(self, sensor_res):
        """ Set sensor resolution.
        
        Args:
            sensor_res (list): sensor resolution.
        """
        self.sensor_res = sensor_res
        self.prepare_sensor(sensor_res)

    def load_file(self, filename,device=DEVICE):
        """ Load lens from .json file.

        Args:
            filename (string): lens file.
        """
        if filename[-5:] == '.json':
            self.read_lens_json(filename,device=device)

        else:
            raise Exception(f"get filename: {filename   }, File format not supported.")

    def load_external(self, surfaces, obj_plane=None, sensor_plane=None,device=DEVICE):
        """ Load lens from extrenal surface/material list.
        """
        self.surfaces = surfaces

        self.obj_plane = obj_plane

        self.r_last = sensor_plane.r
        self.d_sensor = torch.tensor(sensor_plane.d,device=device)
        self.sensor_size = sensor_plane.sensor_size
        
        # find aper_idx
        self.aper_idx = None
        for i, s in enumerate(self.surfaces):
            if isinstance(s, Aperture):
                if self.aper_idx is not None: # if more than one aperture found, use the last one
                    print(f"Warning: More than one aperture found in the lens, using the last one: {self.aper_idx}.")
                self.aper_idx = i
        
        self.device = device
        self.CRF = CRF()

    def prepare_sensor(self, sensor_res=[512, 512], sensor_size=None):
        """ Create sensor. 

            reference values:
                Nikon z35 f1.8: diameter = 1.912 [cm] ==> But can we just use [mm] in our code?
                Congli's caustic example: diameter = 12.7 [mm]
        Args:
            sensor_res (list): Resolution, pixel number.
            pixel_size (float): Pixel size in [mm].

            sensor_res: (H, W)
        """
        sensor_res = [sensor_res, sensor_res] if isinstance(sensor_res, int) else sensor_res
        self.sensor_res = sensor_res
        H, W = sensor_res
        if sensor_size is None:
            self.sensor_size = [2 * self.r_last * H / np.sqrt(H**2 + W**2), 2 * self.r_last * W / np.sqrt(H**2 + W**2)]
        else:
            self.sensor_size = sensor_size
            self.r_last = np.sqrt(sensor_size[0]**2 + sensor_size[1]**2) / 2
        self.sensor_size = [float(x) for x in self.sensor_size] # convert to float

        assert self.sensor_size[0] / self.sensor_size[1] == H / W, "Pixel is not square."
        self.pixel_size = self.sensor_size[0] / sensor_res[0]

    # ====================================================================================
    # Ray Sampling
    # ====================================================================================
        
    @torch.no_grad()
    def sample_ray_angle_1D(self, 
                            angle:float = 0, 
                            depth:Union[float, str] = "inf", 
                            M:int = 8, 
                            wvln:float = DEFAULT_WAVE, 
                            r_range:list = [0,1],
                            axis="x"
                            ) -> Ray:
        """ Sample forward rays accoring to specifict field angle. Rays hape shape of [M, 3]. Notice this samples on x axis, so y is always 0.

            Used for (1) drawing lens setup, (2) paraxial optics calculation

        Args:
            angle (float): incident angle (in degree). Defaults to 0.
            depth (float or str): sampling depth. Default is "inf", which means the rays are parallel. 
            M (int): ray number. Defaults to 9.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.
            r_range (list): range of the pupil radius. Defaults to [0,1], which means the whole pupil.
            axis (str): axis to sample on. Defaults to "x". 

        Returns:
            ray (Ray object): Ray object with points shape [M, 3].
        """
        dim = 0 if axis=="x" else 1
        
        # Second point on the pupil
        pupilz, pupilx = self.entrance_pupil()
        pts_pupil_xy = sample_pts_pupil_1D(spp=M, axis=axis, r_range=r_range).to(self.device) * pupilx # [M, 2]
        pts_pupil_z = torch.ones_like(pts_pupil_xy[...,:1]) * pupilz # z coordinate is the pupil depth
        o2 = torch.cat([pts_pupil_xy, pts_pupil_z], dim=-1) # [M, 3]

        if depth == "inf":
            d = torch.zeros((M, 3), device=self.device)
            d[:,dim] = np.sin(angle / 57.3)
            d[:,2] = np.cos(angle / 57.3)
            o1 = o2 # First point on the entrance pupil
        else:
            if depth == None:
                depth = self.obj_plane.d
            assert type(depth) in [int, float, torch.Tensor], f"depth should be a number or 'inf'. now get depth={depth}."
            # First point is the point source
            o1 = torch.zeros((M, 3), device=self.device)
            o1[:, dim] = depth * np.tan(angle / 57.3)
            o1[:, 2] = depth
            # Form the rays and propagate to z = 0
            d = o2 - o1
            
        ray = Ray(o1, d, wvln, device=self.device)
        ray.propagate_to(z=self.surfaces[0].d - 0.1)    # ray starts from z = - 0.1

        return ray

    @torch.no_grad()
    def sample_ray_render(self, spp=64, wvln=DEFAULT_WAVE,roi_pix=None,pupil_size=None,random_pupil=True):
        """ Ray tracing rendering step1: sample ray and go through lens.
        Sample rays from sensor pixels. Rays have shape of [spp, H, W, 3].

        Args:
            spp (int): sample per pixel. Defaults to 1.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.
            roi_pix (list or ndarray): region of interest in pixel coordinate. Defaults to None. [x,y,w,h]
            pupil_size (float): size of the exit pupil radius. Defaults to None, which means using the default pupil size.
            random_pupil (bool): whether to sample points randomly on the pupil. Defaults to True.
        """
        # ===> sample o1 on sensor plane
        if roi_pix is None:
            h,w = self.sensor_res
        else:
            x,y,w,h = roi_pix
        o1 = self.sample_pts_sensor(grid_size=(h,w), roi=roi_pix, roi_type='pix', distribution='uniform', align_corner=False, flip=True) # [h, w, 3]
        
        # ==> Sample o2 on the second plane and compute rays
        pupilz, pupilr = self.exit_pupil()
        if pupil_size is not None:
            pupilr = pupil_size
        pts_pupil_xy = sample_pts_pupil_grid(grid_size=[h,w],spp=spp,random=random_pupil).to(self.device)*pupilr # [spp, h, w, 2]
        pts_pupil_z = torch.ones_like(pts_pupil_xy[...,:1]) * pupilz # z coordinate is the pupil depth
        o2 = torch.cat([pts_pupil_xy, pts_pupil_z], dim=-1) # [spp, h, w, 3]
        o1 = torch.broadcast_to(o1, o2.shape)
        d = o2 - o1    # broadcast to [spp, h, w, 3]
            
        ray = Ray(o1, d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_ray_points2pupil(self, points, pupil="entrance", pupil_scale=1.0, spp=256, wvln=DEFAULT_WAVE):
        """ Sample forward rays from given point source (un-normalized positions). Rays have shape [spp, N, 3]
            Used for (1) PSF calculation, (2) chief ray calculation.

        Args:
            points (list): ray origin of shape [N,3].
            pupil: "entrance" or "exit", which pupil to sample from.
            pupil_scale (float): scale of the pupil radius. Defaults to 1.
            spp (int): sample per pixel. Defaults to 8.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray: Ray object. Shape [spp, N, 3]
        """
        # 1. construct origin of shape [spp, N, 3]
        assert len(points.shape) == 2 and points.shape[1] == 3, "o should be of shape [N,3], now get shape {}".format(points.shape)
        N,_ = points.shape
        o = points.unsqueeze(0).repeat(spp, 1, 1) # [spp, N, 3]
        
        # 2. Sample on pupil and compute direction of shape [spp, N, 3]
        if pupil == "entrance":
            pupilz, pupilr = self.entrance_pupil()
        elif pupil == "exit":
            pupilz, pupilr = self.exit_pupil()
        else:
            raise Exception(f"Unknown pupil type: {pupil}. Available options are 'entrance' or 'exit'.")
        
        # 2. Sample points on the pupil plane
        pts_pupil_xy = sample_pts_pupil_grid(grid_size=[N,],spp=spp,random=True).to(self.device)*(pupilr*pupil_scale) # [spp, N, 2]
        pts_pupil_z = torch.ones_like(pts_pupil_xy[...,:1]) * pupilz # z coordinate is the pupil depth
        o2 = torch.cat([pts_pupil_xy, pts_pupil_z], dim=-1) # [spp, N, 3]

        # 3. construct rays
        d = o2 - o # [spp, N, 3]
        ray = Ray(o, d, wvln, device=self.device)
        return ray

    # ====================================================================================
    # Sample Points
    # ====================================================================================
    
    def sample_pts_sensor(self, grid_size, roi=None, roi_type='pix', distribution='uniform', align_corner=False, flip=True):
        """ Sample points on sensor plane. The output is a tensor of shape [H, W, 3] on real world coordinates.
        
        Args:
            grid_size (tuple): grid size for point sampling. Defaults to (11, 11).
            roi (list): region of interest in pixel coordinates [x,y,w,h]. Defaults to None, which means the whole sensor.
            roi_type (str): 'pix' or 'rel'. Defaults to 'pix'.
            distribution (str): 'uniform' or 'gaussian'. Defaults to 'uniform'.
            align_corner (bool): whether to align the corner of the grid. Defaults to False.
            flip (bool): whether to flip the x and y coordinates. Defaults to True.
        
        Returns:
            pts (tensor): sampled points on sensor plane, shape [H, W, 3]. The last dimension is [x,y,z] in real world coordinates.
        """
        assert roi_type in ['pix', 'rel'], "roi_type should be 'pix' or 'rel'."
        if roi is None:
            if roi_type == 'pix':
                H,W = self.sensor_res
                x,y,w,h = [0, 0, W, H] # sensor_res [H, W]
            elif roi_type == 'rel':
                x,y,w,h = [-1, -1, 2, 2]
        else:
            x,y,w,h = roi   # [x,y,w,h]
        
        scale = torch.tensor([w/2.0,h/2.0]).to(self.device) # scale by roi
        bias = torch.tensor([x+w/2.0,y+h/2.0]).to(self.device) # bias by roi
        
        # sample points on sensor plane
        pts_grid = sample_pts_grid_2D(grid_size=grid_size, align_corner=align_corner, distribution=distribution).to(self.device) # [h, w, 2] x,y range [-1, 1]
        if flip:
            pts_grid = torch.flip(pts_grid, [0,1]) # flip x, y to revert the top-left sampling to bottom-right sampling
            if roi_type == 'pix':
                H,W = self.sensor_res
                WH = torch.tensor([W,H]).to(self.device)
            elif roi_type == 'rel':
                WH = torch.tensor([0,0]).to(self.device)
            bias = WH - bias # flip the bias to revert the top-left sampling to bottom-right sampling

        # convert to img coordinates (physical space) in 2D
        if roi_type == 'pix':
            pts_pix = pts_grid * scale + bias
            pts_img = cvt_pix2img(pts_pix, self.sensor_res, self.sensor_size) 
        elif roi_type == 'rel':
            pts_rel = pts_grid * scale + bias 
            pts_img = cvt_rel2img(pts_rel, self.sensor_size) 
        
        

        pts_z = torch.ones_like(pts_img[...,:1]).to(self.device) * self.d_sensor # z coordinate is the sensor depth
        pts = torch.cat([pts_img, pts_z], 2)
        
        return pts
        
        


    # ====================================================================================
    # Ray Tracing functions
    # ====================================================================================
    def trace_lens(
        self, 
        ray: Ray, 
        lens_range: Union[range,None] = None, 
        record: bool = False
    ) -> tuple[Ray]:
        """
        General ray tracing function, only trace within the lenses

        Args:
            ray (Ray): Input ray object to be traced through the lens system.
            lens_range (Union[range,None]): Range of lens surfaces to trace through. If None, traces all surfaces.
            record (bool): If True, records the ray path for visualization.

        Returns:
            Tuple: containing
                - ray_final (Ray): Ray after passing through the optical system.
        """
        
        
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))
            
        # adjust lens_range according to the ray direction
        is_forward = (ray.d.reshape(-1, 3)[0, 2] > 0)
        if is_forward:
            ray.propagate_to(self.surfaces[0].d - 0.1)  # for high-precision opd calculation
        else:
            lens_range = np.flip(lens_range) # reverse the lens range for backward tracing
            
        
        # record the initial ray trace if required
        if record:
            ray.log_trace()  # Log the ray trace for visualization
        
        # iteratively trace through each surface
        for i in lens_range:
            ray = self.surfaces[i].ray_reaction(ray)
            
            if record:
                ray.log_trace()  # Log the ray trace for visualization        

        # if is_forward:
        #     ray_out = self._forward_tracing(ray, lens_range, record=record)
        # else:
        #     ray_out = self._backward_tracing(ray, lens_range, record=record)
            
        return ray

    def trace2obj(self, 
                  ray:Ray, 
                  lens_range: Union[range,None] = None, 
                  record:bool=False,
                  depth:Union[float,None]=None,
                  )-> Ray:
        """
        Ray tracing to object plane or specified depth.

        Args:
            ray (Ray): Input ray object to be traced through the lens system.
            lens_range (Union[range,None]): Range of lens surfaces to trace through. If None, traces all surfaces.
            record (bool): If True, records the ray path for visualization.
            depth (Union[float,None]): If provided, propagates the ray to this depth after tracing. If None, propagates to the object plane.

        Returns:
            Ray: Ray after passing through the optical system and propagated to the specified plane or depth.
        """
        ray = self.trace_lens(ray, lens_range, record)
        if depth is None:
            ray = self.obj_plane.ray_reaction(ray)
        else:
            ray.propagate_to(depth)
            
        if record:
            ray.log_trace()
        return ray
 
    def trace2sensor(self, 
                     ray:Ray, 
                     lens_range: Union[range,None] = None, 
                     record:bool=False
                    )-> Ray:
        """
        Ray tracing to the sensor plane.

        Args:
            ray (Ray): Input ray object to be traced through the lens system.
            lens_range (Union[range,None]): Range of lens surfaces to trace through. If None, traces all surfaces.
            record (bool): If True, records the ray path for visualization.

        Returns:
            Ray: Ray after passing through the optical system and propagated to the sensor plane.
        """
        ray = self.trace_lens(ray, lens_range, record)
        ray = ray.propagate_to(self.d_sensor)
        if record:
            ray.log_trace()  # Log the ray trace for visualization

        return ray

    def proj_sensor2obj(self, pts_sensor, depth=None):
        """ Compute reference PSF center (flipped, green light) for given point source.

        Args:
            pts_sensor: [N, 3] real-world unit (mm) point is in sensor plane.

        Returns:
            obj_point: [N, 3] real-world unit (mm) point in object plane.
            valid: [N] valid ray number.
        """
        # Shrink the pupil and calculate centroid ray as the chief ray. Distortion is allowed.
        ray = self.sample_ray_points2pupil(pts_sensor, pupil="exit", spp=GEO_SPP, pupil_scale=0.1)
        ray = self.trace2obj(ray,depth=depth)
        assert (ray.ra == 1).any(), 'No sampled rays is valid.'
        obj_point = (ray.o * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON) # shape [N, 3], take the centroid of the rays
        valid = ray.ra.sum(0) # to shape [N]
        return obj_point, valid>0

    def proj_obj2sensor(self, pts_obj):
        """ Compute reference PSF center (flipped, green light) for given point source.

        Args:
            pts_obj: [N, 3] real-world unit (mm) point in sensor plane.

        Returns:
            sensor_point: [N, 3] real-world unit (mm) point is in object plane.
            valid: [N] valid ray number.
        """
        # Shrink the pupil and calculate centroid ray as the chief ray. Distortion is allowed.
        ray = self.sample_ray_points2pupil(pts_obj, pupil="entrance", spp=GEO_SPP, pupil_scale=0.1) # ray.o has shape (spp, N, 3)
        ray = self.trace2sensor(ray)
        assert (ray.ra == 1).any(), 'No sampled rays is valid.'
        sensor_point = (ray.o * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON) # shape [N, 3], take the centroid of the rays
        valid = ray.ra.sum(0)

        return sensor_point,valid > 0

        
    # ====================================================================================
    # Ray-tracing based rendering
    # ====================================================================================
    
    def disconnect(self):
        """ Disconnect the hardware if exsist.
        """
        for surface in self.surfaces:
            surface.disconnect()

    def update(self, fix_surfaces=[],clip=True):
        """ Update depth among surfaces and update parameters for each surface.

        Args:
            fix_surfaces (list): list of surface idx that are fixed in location, otherwise, they will be updated based on the previous surface.
        """
        move_srufaces = list(range(1,len(self.surfaces)))
        for i in fix_surfaces:
            move_srufaces.remove(i)
        # Update depth of all surfaces of the lens according to the previous surface if itself don't have differentiable "d".
        for i in move_srufaces:
            if "d" not in self.surfaces[i].dif_able:
                self.surfaces[i].d = self.surfaces[i-1].d + self.d_next_list[i-1]
            if "n" not in self.surfaces[i].dif_able:
                # same lens should share same normal
                if type(self.surfaces[i]) == Spheric and type(self.surfaces[i]) == Spheric:
                    self.surfaces[i].n = self.surfaces[i-1].n
        
        # # correcting distances such that the first surface is at d=0
        # with torch.no_grad():
        #     if self.surfaces[0].d != 0:
        #         d_start = self.surfaces[0].d.item()
        #         for i in range(0,len(self.surfaces)):
        #             self.surfaces[i].d -= d_start
        #         self.obj_plane.d -= d_start
        #         self.d_sensor -= d_start
 
        for surface in self.surfaces:
            # update parameters for each surface if implemented
            if type(surface) == DPP:
                surface.update(clip=clip)
            else:
                surface.update()
        
        self.correct_shape()
        
    def render(self, img=None, spp=64, method='ray_tracing',roi_pix=None, sensor_res=None):
        """ This function is defined for End-to-End lens design. It simulates the camera-captured image batch and it is differentiable.

            2 kinds of rendering methods are supported:
                1. ray tracing based rendering
                2. PSF based rendering

        Args:
            img (tensor): [N, C, H, W] shape image batch.
            spp (int, optional): sample per pixel. Defaults to 64.
            method (str, optional): rendering method. Defaults to 'ray_tracing'.
            roi_pix (list, optional): region of interest in pixel coordinates [x,y,w,h]. Defaults to None, which means the whole sensor.
            sensor_res (list, optional): sensor resolution. Defaults to None, which means using the current sensor resolution.

        Returns:
            img_render (tensor): [N, C, H, W] shape rendered image batch.
        """
        # ==> Prepare sensor resolution
        prev_sensor_res = self.sensor_res
        if sensor_res is None:
            sensor_res = self.sensor_res
        
        self.prepare_sensor(sensor_res=sensor_res)
        
        if roi_pix is None:
            x,y,w,h = [0, 0, sensor_res[1], sensor_res[0]] # sensor_res [H, W]
        else:
            x,y,w,h = roi_pix   # [x,y,w,h]

        if method == 'ray_tracing':
            N,C,_,_ = img.shape
            img_render = torch.zeros((N,C,sensor_res[0],sensor_res[1])).to(self.device)
            # img = torch.flip(img, [-2, -1]) # reverse the order of image to project an inverse image
            for i in range(3):
                ray = self.sample_ray_render(spp=spp, wvln=WAVE_RGB[i],roi_pix=roi_pix)
                ray = self.trace2obj(ray) 
                img_render[:,i,y:y+h,x:x+w] = self.render_compute_image(img[:,i,:,:], ray)

        elif method == 'pinhole':
            N,C,_,_ = img.shape
            img_render = torch.zeros((N,C,sensor_res[0],sensor_res[1])).to(self.device)
            for i in range(3):
                ray = self.sample_ray_render(spp=1, wvln=WAVE_RGB[i],roi_pix=roi_pix,pupil_size=0.01) # sample principal ray
                ray = self.trace2obj(ray) 
                img_render[:,i,y:y+h,x:x+w] = self.render_compute_image(img[:,i,:,:], ray)

        elif method == 'depth':
            ray = self.sample_ray_render(spp=1, wvln=DEFAULT_WAVE,roi_pix=roi_pix,pupil_size=0.01) # sample principal ray on object
            ray = self.trace2obj(ray)
            depth = ray.o[...,2].mean(axis=0)
            depth_min, depth_max = depth[ray.ra[0]>0].min(), depth[ray.ra[0]>0].max()
            # normalize depth for visulization
            depth = (depth - depth_min) / (depth_max - depth_min)
            img_render = depth.reshape(1,1,sensor_res[0],sensor_res[1]).repeat(1,3,1,1)
        
        elif method == 'depth_in_focus':
            ray = self.sample_ray_render(spp=spp, wvln=DEFAULT_WAVE,random=True,roi_pix=roi_pix) # sample ray
            ray = self.trace_lens(ray) # (spp, H, W, 3)

            ray_ref = self.sample_ray_render(spp=1, wvln=DEFAULT_WAVE,pupil_size=0,random=False,roi_pix=roi_pix) # sample principal ray
            ray_ref = self.trace_lens(ray_ref) # (1, H, W, 3)

            depth = torch.ones((1, sensor_res[0], sensor_res[1])).to(self.device)*-600 # initial depth at -600 mm
            depth.requires_grad_(True)
            optimizer = torch.optim.Adam([depth], lr=1)
            median_pool2d = MedianPool2d(kernel_size=5, stride=1,padding=2)
            for i in range(2000):
                t = (depth - ray.o[...,2])/ray.d[...,2]
                t_ref = (depth - ray_ref.o[...,2])/ray_ref.d[...,2]

                # calculate the intersection point
                p = ray.o + ray.d * t.unsqueeze(-1)
                p_ref = ray_ref.o + ray_ref.d * t_ref.unsqueeze(-1)

                optimizer.zero_grad()
                psf_r = torch.linalg.norm(p - p_ref, ord=2, dim=-1) # shape [spp, H, W]
                # psf_mean = torch.exp(torch.log(psf_r+EPSILON).sum(0) / ray.ra.sum(0).add(EPSILON)) # shape [H, W]
                psf_mean = psf_r.mean(0)
                loss = torch.mean(psf_mean)
                # loss = torch.sum((p - p_ref)**2)
                loss.backward()
                optimizer.step()
                # logging
                if i % 200 == 0:
                    with torch.no_grad():
                        # median filter
                        depth += median_pool2d(depth.unsqueeze(0)).squeeze() - depth
                    print(f'Iter {i}, Loss {loss.item()}, mean depth {depth.mean().item()}')

            depth = depth.detach().cpu() # (1, H, W)
            psf_mean = psf_mean.detach().cpu().unsqueeze(0) # (1, H, W)
            img_render = torch.cat([depth, psf_mean], dim=0)# (2, H, W)
            img_render = img_render.reshape(1,2,sensor_res[0],sensor_res[1]) # (1, 2, H, W)
            # img_render = depth.reshape(1,1,sensor_res[0],sensor_res[1]) # (1, 1, H, W)

            # # normalize depth for visulization
            # depth_min, depth_max = depth[ray.ra[0]>0].min(), depth[ray.ra[0]>0].max()
            # depth = (depth - depth_min) / (depth_max - depth_min)
            # img_render = depth.reshape(1,1,sensor_res[0],sensor_res[1]).repeat(1,3,1,1)
        
        else:
            raise Exception('Unknown method.')
        
        # img_render = self.CRF(img_render)

        # ==> Change the sensor resolution back
        self.prepare_sensor(sensor_res=prev_sensor_res)
        return img_render

    def render_compute_image(self, img, ray, noise=0):
        """ Ray tracing rendering step2: ray and texture plane intersection and computer rendered image.

            With interpolation. Can either receive tensor or ndarray

            This function receives [spp, W, H, 3] shape ray, returns [W, H, 3] shape sensor output.

            backpropagation, I -> w_i -> u -> p -> ray
        """
        # ====> Preparetion
        if torch.is_tensor(img):    # if img is [N, C, H, W] or [N, H, W] tensor, what situation will [N, H, W] occur?
            H, W = img.shape[-2:]
            if len(img.shape) == 4:
                img = F.pad(img, (1,1,1,1), "replicate")    # we MUST use replicate padding.
            else:
                img = F.pad(img.unsqueeze(1), (1,1,1,1), 'replicate').squeeze(1)

        elif isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img / 255.0
                img = img.astype(np.float32)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            H, W = img.shape[:2]
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
            img = F.pad(img, (1,1,1,1), "replicate")


        # ====> Monte carlo integral
        ray_z = ray.o[...,2] # [spp, H, W] markdown the ray_z in original space

        # Convert back to uv coordinates
        ray_loc = self.obj_plane.glo2loc(ray) # it will change the ray!!!
        pts_obj = self.obj_plane.loc2uv(ray_loc.o)  # u,v is the ray intersection point on the image plane in mm, in object image plane space
        # scale the uv according to the ray_z, ONLY USED FOR DEPTH RENDERING
        if type(self.obj_plane) == DepthPlane:
            scale = ray_z / self.obj_plane.d
            pts_obj = pts_obj / scale # devide scale, effectively magnify the image
            # u,v = u / scale, v / scale # devide scale, effectively magnify the image
        
        # convert to pixel coordinates
        obj_plane_size = (self.obj_plane.h, self.obj_plane.w) # image plane size in mm, h is the height, w is the width
        pts_pix = cvt_img2pix(pts_obj, (H,W), obj_plane_size ) # convert to pixel coordinates
        u,v = pts_pix[...,0], pts_pix[...,1] # u,v is the pixel coordinates of the image, u, v roughly in [0, W] and [0, H], but not clamped yet
        u,v = torch.clamp(u-0.5, min=0, max=W-1.001), torch.clamp(v-0.5, min=0, max=H-1.001) # clamp to [0,W-1) and [0,H-1)
        
        idx_i = v.floor().long() # idx_i is the row index in image space, range [0, H-2]
        idx_j = u.floor().long() # idx_j is the column index in image space, range [0, W-2]
        
        w_i = v.ceil() - v # w_i is the weight for bilinear interpolation in y direction, range [0, 1]
        w_j = u.ceil() - u # w_j is the weight for bilinear interpolation in x direction, range [0, 1]
        
        # Bilinear interpolation
        # img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], irr_img shape [N, C, spp, H, W]
        irr_img =  img[...,idx_i, idx_j] * w_i * w_j
        irr_img += img[...,idx_i+1, idx_j] * (1-w_i) * w_j
        irr_img += img[...,idx_i, idx_j+1] * w_i * (1-w_j)
        irr_img += img[...,idx_i+1, idx_j+1] * (1-w_i) * (1-w_j)

        # I = (torch.sum(irr_img * ray.ra, -3) + 1e-9) / (torch.sum(ray.ra, -3) + 1e-6)
        N,spp,H,W = irr_img.shape
        I = (torch.sum(irr_img * ray.ra, -3) + 1e-9) / spp

        # ====> Add sensor noise
        if noise > 0:
            I += noise * torch.randn_like(I).to(self.device)
        
        # I = torch.flip(I, [-2, -1]) # flip the image to get the correct upside-down image
        return I


    # ====================================================================================
    # PSF and spot diagram
    #   1. Incoherent functions
    #   2. Coherent functions 
    # ====================================================================================
    
    def psf_map(self, 
            grid: Union[int, list] = 7, 
            ks: int = 51, 
            spp: int = GEO_SPP, 
            roi_rel: list = [-1,-1,2,2], 
            stats_only: bool = False,
            align_corner: bool = False, 
            distribution: str = 'uniform',
            pts_sensor: Union[torch.Tensor,None] = None,
            sep_rgb: bool = False
            ):
        """ Compute RGB PSF map at a given depth (or object plane).

            Now used for (1) generate PSFs for PSF-based rendering, (2) draw PSF map

        Args:
            grid ([int,list]): Grid size. Defaults to 7.
            ks (int, optional): Kernel size. Defaults to 51.
            align_corner (bool, optional): Sample corner of the sensor plane. Defaults to False.
            spp (int, optional): Sample per pixel. Defaults to None.
            stats_only (bool, optional): Only return the MSE of the point source. Defaults to False.
            pts_sensor (Tensor, optional): Points on the sensor plane. If None, sample points on the sensor plane. Defaults to None.
            roi_rel (list, optional): Region of interest in relative coordinates. Defaults to [-1,-1,2,2].
            sep_rgb (bool, optional): Whether to separate RGB PSF using different center. Defaults to False.

        Returns:
            (N = grid*grid)
            psf: Shape of [N, ks, ks, 3]. PSF map of sampled points. 
            ptc_sensor: Shape of [N, 2, 3]. position of the PSF center (xy in mm) on the sensor plane. 
            pts_stats: dict, statistics of the point source. elements are [N, 3] or [spp, N, 3] in shape.
            valid_index: valid index of the points, shape of [N_valid]
        """
        with torch.no_grad():
            if pts_sensor is None:
                grid_size = [grid,grid] if isinstance(grid, int) else grid
                # sample points on sensor plane, starting from top-left corner to bottom-right corner
                pts_sensor = self.sample_pts_sensor(grid_size,roi=roi_rel, roi_type="rel", align_corner=align_corner, distribution=distribution, flip = True) # shape [grid, grid, 3]
            else:
                assert pts_sensor.shape[-1] == 3, "pts_sensor should be of shape [N, 3]."
            
            pts_sensor = pts_sensor.reshape(-1, 3)
            
            # project points to object space
            if type(self.obj_plane) == Plane:
                depth = self.obj_plane.d
            elif type(self.obj_plane) == DepthPlane:
                depth = None
            else:
                raise Exception(f"Unknown object plane type: {type(self.obj_plane)}. Only Plane and DepthPlane are supported.")
            pts_obj,valid = self.proj_sensor2obj(pts_sensor,depth=depth)
            N, _ = pts_obj.shape
            
            # only run on valid_indices to save computations
            valid_index = torch.where(valid)[0]
            pts_obj_valid = pts_obj[valid_index] # shape [N_valid, 3]
        
        psfs_valid, ptc_sensor_valid,pts_stats = self.psf_rgb(pts_obj=pts_obj_valid, ks=ks, spp=spp,stats_only=stats_only,sep_rgb=sep_rgb, flip = True) # of shape [N_valid, 3, ks, ks]
        
        if valid.all():
            psfs = psfs_valid
            ptc_sensor = ptc_sensor_valid
        else:
            # construct output with original N = grid**2 from valid_index
            psfs = torch.zeros((N, ks, ks, 3), device=pts_obj.device)
            ptc_sensor = torch.zeros((N, 2, 3), device=pts_obj.device) # [N, 2, 3] for x,y position on the sensor plane (mm)
            psfs[valid_index] = psfs_valid
            ptc_sensor[valid_index] = ptc_sensor_valid

            # update stat with max value if invalid
            for k,v in pts_stats.items():
                v_shape = list(v.shape)
                if k in ['raw','ra']:
                    v_shape[1] = N
                    stat = torch.ones((v_shape), device=pts_obj.device) * v.max() # [spp, N, 3] use max value to fill the invalid points
                    stat[:,valid_index] = v
                else:
                    v_shape[0] = N
                    stat = torch.ones((v_shape), device=pts_obj.device) * v.max() # [N, 3] use max value to fill the invalid points
                    stat[valid_index] = v
                
                pts_stats[k] = stat

        return psfs, ptc_sensor, pts_stats, valid_index
        
        
    def psf_rgb(self, pts_obj, ks=31, spp=GEO_SPP, stats_only=False,sep_rgb=False,flip=True):
        """ Compute RGB point PSF. This function is differentiable.
        
        Args:
            pts_obj (torch.Tensor): Shape of [N, 3], point is in object space, normalized.
            ks (int): Output kernel size. Defaults to 7.
            spp (int): Sample per pixel. Defaults to 2048.
            stats_only (bool): Only return the MSE of the point source. Defaults to False.
            sep_rgb (bool): Whether to separate RGB PSF using different center. Defaults to False.
            flip (bool, optional): Whether to flip the x and y coordinates of integrated PSF. Defaults to True.

        Returns:
            psf: Shape of [N, ks, ks, C] or [ks, ks, C].
            ptc_sensor: Shape of [N, 2, C]. position of the PSF center on the sensor plane.
            
        """
        psfs = []
        ptcs = [] # point centers on the sensor plane 
        pts_stats = {}
        if sep_rgb: 
            ptc_sensor = None
        else: # use the same PSF center for all RGB channels
            ptc_sensor,valid = self.proj_obj2sensor(pts_obj) # shape [N, 3]
        for wvln in WAVE_RGB:
            psf,ptc,point_stats = self.psf_wvln(pts_obj=pts_obj, ptc_sensor=ptc_sensor, wvln=wvln, ks=ks, spp=spp,stats_only=stats_only,flip=flip) # psf shape of [N, ks, ks], ptc shape of [N, 3]
            psfs.append(psf)
            ptcs.append(ptc)
            for k,v in point_stats.items():
                if k not in pts_stats:
                    pts_stats[k] = []
                pts_stats[k].append(v)
        
        psf = torch.stack(psfs, dim =-1) # [N, ks, ks, 3] or [ks, ks, 3] if single point
        ptc_sensor = torch.stack(ptcs, dim=-1) # [N, 2, 3] or [2, 3] if single point
        for k,v in pts_stats.items():
            pts_stats[k] = torch.stack(v, dim=-1) # [N, 3] (or [3] if single point)

        return psf, ptc_sensor, pts_stats

    def psf_wvln(self, pts_obj, ptc_sensor=None, ks=7, wvln=DEFAULT_WAVE, spp=GEO_SPP,stats_only=False,flip=True):
        """ Single wvln incoherent PSF calculation.

        Args:
            pts_obj (Tnesor): Point source position in real-world unit (mm). Shape of [N, 3].
            ptc_sensor (Tensor,optional): if given Reference PSF center in real-world unit (mm). Shape of [N, 2].
            ks (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. For diff ray tracing, usually kernel_size^2. Defaults to 2048.
            stats_only (bool, optional): Only return the MSE of the point source. Defaults to False.
            flip (bool, optional): Whether to flip the x and y coordinates of integrated PSF. Defaults to True.

        Returns:
            psf: Shape of [N, ks, ks] or [ks, ks]. PSF of the point source.
            ptc_sensor: Shape of [N, 2]. position of the PSF center on the sensor plane.
            point_rms: Shape of [N]. RMS of the point source on the sensor plane.
        """
        # Points shape of [N, 3]
        if not torch.is_tensor(pts_obj):
            pts_obj = torch.tensor(pts_obj)
        if len(pts_obj.shape) == 1:
            single_point = True
            pts_obj = pts_obj.unsqueeze(0)
        else:
            single_point = False
        
        # Trace rays to sensor plane
        # ray = self.sample_ray_from_points(o=pts_obj, spp=spp, wvln=wvln)
        ray = self.sample_ray_points2pupil(pts_obj, pupil="entrance", spp=spp, pupil_scale=1.0,wvln=wvln) # full pupil size sample for PSF
        ray = self.trace2sensor(ray)

        # determin the PSF center
        if ptc_sensor is None:
            # PSF center on the sensor plane by chief ray projection 
            ptc_sensor,valid = self.proj_obj2sensor(pts_obj) # shape [N, 3]
        else:
            # use external PSF center on the sensor plane, usually used for consistency
            pass

        # Calculate PSF
        psf, ptc_sensor, pts_stats = forward_integral(ray, ps=self.pixel_size, ks=ks, pointc_ref=ptc_sensor[...,:2],stats_only=stats_only)
        
        # Normalize to 1
        psf = psf / psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        
        if single_point:
            psf = psf.squeeze(0)
            
        if flip:
            psf = torch.flip(psf, [-2, -1]) # flip the PSF to get the correct orientation

        return psf, ptc_sensor, pts_stats

    
    # ====================================================================================
    # Geometrical optics 
    #   1. Focus-related functions
    #   2. FoV-related functions
    #   3. Pupil-related functions
    # ====================================================================================

    # ---------------------------
    # 1. Focus-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_foclen(self, depth="inf", wvln=DEFAULT_WAVE, M=10, return_fig=False, r_range=[0.05,0.2], color="r"):
        """ Compute focal length (FL): distance between the second principal point and the in-focus position. This is essentially the RFL: Rear focal length. 
            When the optical system is in air, we have RFL = FFL = EFL (Rear/Front/Equivalent Focal Length). Ref: https://en.wikipedia.org/wiki/Focal_length
            
        Args:
            wvln ([Optional(float)]): Wavelength of light. Defaults to DEFAULT_WAVE
            depth (float or str, optional): Depth of the point source plane. Defaults to "inf", which means the lens is focused at infinity.
            M (int, optional): Number of rays to sample. Defaults to 10.
            return_fig (bool, optional): Whether to return the figure. Defaults to False.
            ring (list, optional): Ring radius range for paraxial sampling rays. Defaults to [0.05,0.2].   
            color (str, optional): Color of the plot. Defaults to "r".
            
        Returns:
            Tuple:
            - rfl (float): Rear focal length in mm.
            - out_dict (dict): Dictionary containing z_principal and z_focus.
            - If return_fig is True, it also contains the figure.
        """
        # Forward ray tracing
        ray = self.sample_ray_angle_1D(angle=0,depth=depth,M=M,wvln=wvln, r_range=r_range)
        inc_ray = ray.clone()
        out_ray = self.trace_lens(ray,record=True)

        # (Rear) Principal point
        t0 = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z_principal = out_ray.o[..., 2] - out_ray.d[..., 2] * t0


        # Focal point
        t1 = - out_ray.o[..., 0] / out_ray.d[..., 0]
        z_focus = out_ray.o[..., 2] + out_ray.d[..., 2] * t1


        # (Rear) Focal Length
        rfl = z_focus - z_principal
        rfl = np.nanmean(rfl[ray.ra > 0].cpu().numpy()).item()

        out_dict = {
                'z_principal': np.nanmean(z_principal[ray.ra > 0].cpu().numpy()).item(), 
                'z_focus': np.nanmean(z_focus[ray.ra > 0].cpu().numpy()).item()
                }
        
        if return_fig:
            ax, fig = self.plot_setup2D(with_sensor=False)
            self.plot_raytraces(out_ray.trace, ax, fig,plot_invalid=True,ra=ray.ra)
            out_z = out_ray.o[...,2].cpu().numpy()
            out_x = out_ray.o[...,0].cpu().numpy()
            inc_x = inc_ray.o[...,0].cpu().numpy()
            ax.plot([z_principal.cpu().numpy(),out_z], [inc_x,out_x], color, linewidth=0.8,linestyle='dashed')
            ax.plot([z_focus.cpu().numpy(),out_z], [np.zeros_like(out_x),out_x], color, linewidth=0.8,linestyle='dashed')
            # plot vertical line on z_principal and z_focus
            z_principal = np.nanmean(z_principal[ray.ra > 0].cpu().numpy())
            z_focus = np.nanmean(z_focus[ray.ra > 0].cpu().numpy())
            ax.axvline(z_principal, color=color, linestyle='dashed', linewidth=0.8)
            ax.axvline(z_focus, color=color, linestyle='dashed', linewidth=0.8)
            # plot vertical line at z=0
            ax.axvline(0, color='k', linewidth=0.8)
            # ax.axis('off')
            ax.set_title(f'Focal Length: {rfl:.2f} mm \n Z_principal: {z_principal:.2f} mm, Z_focus: {z_focus:.2f} mm')
            # # tight layout
            # fig.tight_layout()
            # limit y axis
            ax.autoscale(False)
            ax.set_xlim(-21,70)
            fig.set_tight_layout(True)  # Apply tight layout after setting limits
            
            out_dict['fig'] = fig
        
        return rfl, out_dict

    def calc_eqfl(self, hfov):
        """ 35mm equivalent focal length.
            35mm sensor: 36mm * 24mm. Diagonal length is 43.27mm, half of the diagonal is 21.63mm.
        Args:
            hfov (float): Half diagonal field of view in radians.
        Returns:
            eqfl (float): Equivalent focal length in mm.
        """
        return (21.63 / np.tan(hfov)).item()


    # ---------------------------
    # 2. FoV-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_hfov(self, spp=100, pupil_scale=0.2):
        """ Compute half diagonal fov.

            Shot rays from edge of sensor, trace them to the object space and compute
            angel, output rays should be parallel and the angle is half of fov.
            
            Args:
                pupil_scale (float, optional): Scale of the exit pupil. Defaults to 0.2.
            
            Returns:
                fov (float): Half diagonal field of view in radians.
        """
        # Sample rays going out from edge of sensor, shape [spp, 3] 
        pts_sensor = torch.tensor([self.r_last, 0, self.d_sensor]).repeat(spp, 1)

        # sample points on the exit pupil
        pupilz, pupilx = self.exit_pupil() # limit pupil size to avoid aberations
        pts_pupil_xy = sample_pts_pupil_1D(spp=spp, axis='x',r_range=[0,pupil_scale]) # shape [spp, 2], x is the radius, y is 0
        pts_pupil_z = torch.full_like(pts_pupil_xy[:,:1], pupilz) # z coordinate is constant 
        pts_pupil = torch.cat((pts_pupil_xy, pts_pupil_z), dim=-1) # shape [spp, 3], points on the exit pupil

        ray = Ray(pts_sensor, pts_pupil - pts_sensor, device=self.device)
        ray = self.trace2obj(ray,depth=self.obj_plane.d)

        # compute fov as the averaged outgoing ray angle
        tan_fov = ray.d[...,0] / ray.d[...,2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))
        if torch.isnan(fov):
            print('computed fov is NaN, use 0.5 rad instead.')
            fov = 0.5
        else:
            fov = fov.item()
        
        return fov

    
    # ---------------------------
    # 3. Pupil-related functions
    # ---------------------------

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """Compute the intersection points of 2D lines.

        Args:
            origins (torch.Tensor): Origins of the lines. Shape: [N, 2]
            directions (torch.Tensor): Directions of the lines. Shape: [N, 2]

        Returns:
            torch.Tensor: Intersection points. Shape: [N*(N-1)/2, 2]
        """
        N = origins.shape[0]

        # Create pairwise combinations of indices
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # Shape: [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # Shape: [N*(N-1)/2, 2]
        Di = directions[idx_i]  # Shape: [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # Shape: [N*(N-1)/2, 2]

        # Vector from Oi to Oj
        b = Oj - Oi  # Shape: [N*(N-1)/2, 2]

        # Coefficients matrix A
        A = torch.stack([Di, -Dj], dim=-1)  # Shape: [N*(N-1)/2, 2, 2]

        # Solve the linear system Ax = b
        # Using least squares to handle the case of no exact solution
        x, _ = torch.linalg.lstsq(
            A,
            b.unsqueeze(-1),
        )[:2]
        x = x.squeeze(-1)  # Shape: [N*(N-1)/2, 2]
        s = x[:, 0]
        t = x[:, 1]

        # Calculate the intersection points using either rays
        P_i = Oi + s.unsqueeze(-1) * Di  # Shape: [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # Shape: [N*(N-1)/2, 2]

        # Take the average to mitigate numerical precision issues
        P = (P_i + P_j) / 2
        
        # filter out abnormal value if the value is way larger than the median radius
        median_radius = torch.median(torch.linalg.norm(P, ord=2, dim=-1))
        abnormal_mask = torch.linalg.norm(P, ord=2, dim=-1) > 2 * median_radius
        P = P[~abnormal_mask]

        return P

    @torch.no_grad()
    def exit_pupil(self, aper_idx = None):
        """ Sample **forward** rays to compute z coordinate and radius of exit pupil. 
            Exit pupil: ray comes from sensor to object space. 
        """
        return self.entrance_pupil(entrance=False, aper_idx=aper_idx)
    
    @torch.no_grad()
    def entrance_pupil(self, M=32, entrance=True, aper_idx=None):
        """Sample **backward** rays, return z coordinate and radius of entrance pupil. Entrance pupil: how many rays can come from object space to sensor.

        Reference: https://en.wikipedia.org/wiki/Entrance_pupil "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
        """
        paraxial_scale = 0.2 # scale of the paraxial rays, avoid ray tracing aberrations, usually 0.2 or 0.5

        # Sample M rays from edge of aperture to last surface.
        if aper_idx is None:
            if self.aper_idx is None or hasattr(self, "aper_idx") is False:
                if entrance:
                    return self.surfaces[0].d.item(), self.surfaces[0].r
                else:
                    return self.surfaces[-1].d.item(), self.surfaces[-1].r
            aper_idx = self.aper_idx
            
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r
        
        ray_o = torch.tensor([[aper_r * paraxial_scale, 0, aper_z]]).repeat(M, 1) # shape [M, 3], sample rays from the edge of aperture

        # Sample phi ranges from [-0.5rad, 0.5rad]
        phi = torch.linspace(-0.2, 0.2, M)
        
        if entrance: # entrance pupil, sample ray goes from sensor to object space
            d = torch.stack(
                (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
            )
        else: # exit pupil, sample ray goes from object space to sensor
            d = torch.stack(
                (torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)), axis=-1
            )

        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing
        if entrance: # entrance pupil, lenses in front of aperture forms the image
            lens_range = range(0, aper_idx)
            ray = self.trace_lens(ray, lens_range=lens_range)
        else: # exit pupil, lenses behind aperture forms the image
            lens_range = range(aper_idx + 1, len(self.surfaces))
            ray = self.trace_lens(ray, lens_range=lens_range)

        # Compute intersection points. o1+d1*t1 = o2+d2*t2
        ray_o = ray.o[ray.ra != 0][:,[0,2]]  # filter out invalid rays
        ray_d = ray.d[ray.ra != 0][:,[0,2]]  # filter out invalid rays

        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)
        if len(intersection_points) == 0:
            raise Exception("Cannot compute intersection points, please check the ray tracing results. Maybe the lens is not well defined or the rays are not valid.")
        else:
            avg_pupilx = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

        avg_pupilx = avg_pupilx / paraxial_scale  # scale back to the original size
        return avg_pupilz, avg_pupilx
    
    @torch.no_grad()
    def calc_fnum(self, foclen):
        """ Calculate f-number (focal ratio) of the lens system.
            where fnum = focal length / entrance pupil radius.
            Ref: https://en.wikipedia.org/wiki/F-number
            
        Args:
            foclen (float): Focal length in mm.
        Returns:
            fnum (float): F-number of the lens system.
        """
        
        pupilz, pupilx = self.entrance_pupil()
        fnum = foclen / (2 * pupilx)
        
        return fnum
    
    # ====================================================================================
    # Lens operation 
    #   1. Set lens parameters
    #   2. Lens operation (init, reverse, spherize), will be abandoned
    #   3. Lens pruning
    # ====================================================================================

    # ---------------------------
    # 1. Set lens parameters
    # ---------------------------
    def set_aperture(self, fnum=None, aper_r=None):
        """ Change aperture radius.
        """
        if aper_r is None:
            assert fnum is not None, 'fnum should be given if aper_r is not given.'
            foclen,_ = self.calc_foclen()  # get focal length
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r

    # ---------------------------
    # 2. Lens operation
    # ---------------------------
    def pertub(self):
        """ Randomly perturb all lens surfaces to simulate manufacturing errors. 

        Including:
            (1) surface position, thickness, curvature, and other coefficients.
            (2) surface rotation, tilt, and decenter.
        
        Called for accurate image simulation, together with sensor noise, vignetting, etc.
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].perturb()

    def double(self):
        """ Use double-precision for the lens group.
        """
        for surf in self.surfaces:
            surf.double()


    # ---------------------------
    # 3. Lens pruning
    # ---------------------------
    @torch.no_grad()
    def prune_surf(self, expand_surf=0.2):
        """Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            expand_surf (float): extra height to reserve.
                - For cellphone lens, we usually use 0.1mm or 0.05 * r_sensor.
                - For camera lens, we usually use 0.5mm or 0.1 * r_sensor.
        """
        # surface range should exclude the aperture surface and dpp surface if exists.
        surface_range = list(range(len(self.surfaces)))
        surface_range.remove(self.aper_idx)
        if self.dpp_idx is not None:
            surface_range.remove(self.dpp_idx)

        # Sample full-fov rays to compute valid surface height
        fov = self.calc_hfov()  # half diagonal fov in radians
        
        ray = self.sample_ray_angle_1D(angle=fov * 57.3, depth=None, M=20)
        ray_out = self.trace2sensor(ray=ray, record=True)
        # ray_trace has shape [N+2,spp, 3], where N is the number of surfaces, including ray origin and sensor plane.
        
        ray_trace = torch.stack(ray_out.trace) # [N+2,spp, 3]
        ray_x_record = ray_trace[:,:,0] # [N+2, spp]
        for i in surface_range:
            # Filter out nan values and compute the maximum height
            valid_heights = ray_x_record[i + 1].abs()
            valid_heights = valid_heights[~torch.isnan(valid_heights)]
            if len(valid_heights) > 0:
                max_ray_height = valid_heights.max().item()
                r_max = self.surfaces[i].get_r_max()
                self.surfaces[i].r = min(r_max,max(4.0, max_ray_height * (1 + expand_surf)))  # expand the surface height
            else:
                print(
                    f"No valid rays for surface {i}, keep the surface unchanged."
                )            

    @torch.no_grad()
    def regulate_self_intersec(self, gap_min=0.1):
        """ Regulate the lens such that the lenses are not intersecting with each other.
        """
        move_list = []
        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i+1]

            r = torch.linspace(0.0, 1.0, 20).to(self.device) * current_surf.r
            z_front = current_surf.sag(r, 0) + current_surf.d
            z_next = next_surf.sag(r, 0) + next_surf.d
            
            dist_min = torch.min(z_next - z_front)
            move_list.append(max(gap_min - dist_min, 0.0))
        
        move_list = torch.tensor(move_list, device=self.device)
        move_list = torch.cumsum(move_list, dim=0)  # cumulative sum to ensure no intersection
        
        for i in range(len(self.surfaces) - 1):
            self.surfaces[i+1].d.data += move_list[i]  # move the next surface 

    @torch.no_grad()
    def correct_shape(self):
        """ Correct wrong lens shape during the lens design.
        """
        # # ==> Rule 1: Move the first surface to z = 0
        # move_dist = self.surfaces[0].d.item()
        # for surf in self.surfaces:
        #     surf.d -= move_dist
        # self.d_sensor -= move_dist

        
        # ==> Rule 2: move surfaces to avoid intersection
        # self.regulate_self_intersec(gap_min=0.1)

        # ==> Rule 4: Prune all surfaces
        # self.prune_surf()


    # ====================================================================================
    # Visualization.
    # ====================================================================================
    @torch.no_grad()
    def analysis(self, 
                 fig_name='./test', 
                 render=False, 
                 multi_plot=False, 
                 plot_invalid=True, 
                 depth=DEPTH, 
                 render_unwarp=False, 
                 lens_title=None,
                 roi_rel=[-1,-1,2,2],
                 plot_dpp=True,
                 ):
        """ Analyze the optical lens.
        """
        # parse fig_name
        save_dir, save_file = os.path.split(fig_name)
        os.makedirs(save_dir, exist_ok=True)

        # Draw lens geometry and ray path
        self.plot_setup2D_with_trace(fig_name=os.path.join(save_dir,f"setup2D_{save_file}.png"),
                                        plot_invalid=plot_invalid, lens_title=lens_title, depth=self.obj_plane.d)

        # Draw spot diagram and PSF map
        self.draw_psf_map(fig_name=os.path.join(save_dir,f"psf_ROI{roi_rel}_{save_file}.png"),grid=13, ks=21,roi_rel=roi_rel,ruler_len=100,text_height=10)
        if roi_rel != [-1,-1,2,2]:
            self.draw_psf_map(fig_name=os.path.join(save_dir,f"psf_global_{save_file}.png"),grid=13, ks=51,roi_rel=[-1,-1,2,2],ruler_len=500,text_height=30)
        
        if plot_dpp:
            self.plot_dpp_zern(save_dir,save_file)
            if self.ftl_idx is not None:
                ftl_zern = self.surfaces[self.ftl_idx].zern
                ftl_zern_amp = self.surfaces[self.ftl_idx].get_zern_amp()
                ftl_zern.plot_zern_sag(params=ftl_zern_amp, fig_name=os.path.join(save_dir,f"ftl_sag_{save_file}"))
            
        # # Calculate RMS error
        # rms_avg, rms_radius_on_axis, rms_radius_off_axis = self.analysis_rms()
        # print(f'On-axis RMS radius: {round(rms_radius_on_axis.item()*1000,3)}um, Off-axis RMS radius: {round(rms_radius_off_axis.item()*1000,3)}um, Avg RMS spot size (radius): {round(rms_avg.item()*1000,3)}um.')

        # Render an image, compute PSNR and SSIM
        if render:
            img_org = cv.cvtColor(cv.imread(f'./datasets/USAF-1951-512.png'), cv.COLOR_BGR2RGB)
            img_render = self.render_single_img(img_org, spp=128, unwarp=render_unwarp, save_name=os.path.join(save_dir,f'render_{-depth:.0f}_{save_file}'), noise=0.01)

            render_psnr = round(compare_psnr(img_org, img_render, data_range=255), 4)
            render_ssim = round(compare_ssim(img_org, img_render, channel_axis=2, data_range=255), 4)
            print(f'Rendered image: PSNR={render_psnr}, SSIM={render_ssim}')
            
    
    def plot_dpp_zern(self,save_dir,save_file):
        # Draw DPP sag
        if self.dpp_idx is not None:
            dpp_zern = self.surfaces[self.dpp_idx].zern
            dpp_zern_amp = self.surfaces[self.dpp_idx].get_zern_amp(truncated=True)
            dpp_zern.plot_zern_sag(params=dpp_zern_amp, fig_name=os.path.join(save_dir,f"dpp_OPD_{save_file}"))

            # Compare ideal and physical DPP
            plot_zern_coeffs(params=dpp_zern_amp,label = ['dpp',],fig_name=os.path.join(save_dir,f"dpp_zern_{save_file}.png"))
        
    @torch.no_grad()      
    def plot_setup2D_with_trace(self, fig_name=None, views=[0], M=8, depth=None, plot_invalid=True, lens_title=None, axis="x", scale=1.0, **kwargs):
        """ Plot lens setup with rays.
        """
        # ==> Title
        hfov = self.calc_hfov()
        if lens_title is None:
            foclen,_ = self.calc_foclen()
            fnum  = self.calc_fnum(foclen)
            eqfl = self.calc_eqfl(hfov)
            lens_title = f'FoV{round(2*hfov*57.3, 1)}({int(eqfl)}mm eqfl)_F/{round(fnum,2)}_DIAG{round(self.r_last*2, 2)}mm_FL{round(foclen,2)}mm'

        # ==> Plot RGB in one figure
        colors_list = 'rgb'
        if kwargs.get('colors_list') is not None:
            colors_list = kwargs['colors_list']
        H,W = self.sensor_size

        ax, fig = self.plot_setup2D(scale=scale)
        if axis=='x':
            angle_scale = W / (W**2 + H**2)**0.5 * scale
        elif axis=='y':
            angle_scale = H / (W**2 + H**2)**0.5 * scale
        angles = [0, angle_scale * np.rad2deg(hfov)*0.5, angle_scale * np.rad2deg(hfov)*0.99]


        for i, angle in enumerate(angles):
            ray = self.sample_ray_angle_1D(angle, depth, M=M, wvln=WAVE_RGB[2-i],r_range = [0,0.95],axis=axis)
            ray = self.trace2sensor(ray=ray, record=True)
            ax, fig = self.plot_raytraces(ray.trace, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra,axis=axis)

        ax.axis('off')
        ax.set_title(lens_title)
        if fig_name is None:
            # if xlim and ylim are specified:
            if 'xlim' in kwargs:
                ax.set_xlim(kwargs['xlim'])
            if 'ylim' in kwargs:
                ax.set_ylim(kwargs['ylim'])
            plt.show()
        else:
            fig.savefig(fig_name, bbox_inches='tight', format='png', dpi=600)
            plt.close(fig)


    def plot_raytraces(self, ray_trace, ax=None, fig=None, color='b-', p=None, valid_p=None, plot_invalid=True, ra=None, axis="x"):
        """ Plot ray paths.
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D()
        else:
            show = False

        dim = 0 if axis=="x" else 1

        ray_trace = torch.stack(ray_trace,dim=0)  # shape [N, spp, 3], where N is the number of surfaces
        for i, os in enumerate(ray_trace.permute(1, 0, 2)):
            o = os.cpu().detach().numpy()
            z = o[...,2].flatten()
            r = o[...,dim].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    r = np.append(r, p[i,dim])
                    z = np.append(z, p[i,2])

            if plot_invalid:
                ax.plot(z, r, color, linewidth=0.8)
            elif ra[i]>0:
                ax.plot(z, r, color, linewidth=0.8)

        if show: 
            plt.show()
            
        return ax, fig

    def plot_setup2D(self, ax=None, fig=None, color='k', with_sensor=True, fix_bound=False,scale=1.0):
        """ Draw lens setup.
        """
        def plot(ax, z, x, color):
            p = torch.stack((x, torch.zeros_like(x, device=self.device), z), axis=-1)
            p = p.cpu().detach().numpy()
            ax.plot(p[...,2], p[...,0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d.item() - APERTURE_WEDGE_LENGTH, d.item() + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)        

        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5,5))
        else:
            show=False

        if len(self.surfaces) == 1: # if there is only one surface, then it should be aperture
            # draw_aperture(ax, self.surfaces[0], color='orange')
            raise Exception('Only one surface is not supported.')

        else:
            # Draw lens
            for i, s in enumerate(self.surfaces):

                # Draw aperture
                if type(s) == Aperture:
                    draw_aperture(ax, s, color='orange')


                # Draw lens surface
                else:
                    r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                    z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))   # draw surface
                    plot(ax, z, r, color)

            # Connect two surfaces
            for i, s in enumerate(self.surfaces):
                if self.materials[i].name == "air": # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)
                    
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([[r_prev], [r]])).to(self.device)

                    plot(ax, z, x, color)
                    plot(ax, z,-x, color)
                    s_prev = s

            # Draw sensor
            if with_sensor:
                d_senor = self.d_sensor.item()
                W = self.sensor_size[1] * scale
                ax.plot([d_senor, d_senor], [-W/2, W/2], color)
        
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        
        if fix_bound:
            ax.set_aspect('equal')
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect('equal', adjustable='datalim', anchor='C') 
            ax.minorticks_on() 
            ax.set_xlim(-0.5, 7.5) 
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig

    
    @torch.no_grad()
    def draw_psf_map(self, 
            grid=11, 
            ks=51, 
            log_scale=False, 
            fig_name=None, 
            roi_rel=[-1,-1,2,2],
            align_corner=False,
            distribution='uniform',
            ruler_len=500,
            text_height=30,
            compact=True
            ):
        """ Draw RGB PSF map at a certain depth. Will draw M x M PSFs, each of size ks x ks.
        """
        # Calculate PSF map
        psfs,ptc_sensor,pts_stats,valid_index = self.psf_map(grid=grid, ks=ks, spp=GEO_SPP,roi_rel=roi_rel,align_corner=align_corner,distribution=distribution) # psfs: [N_valid,3,ks,ks], ptc_sensor: [N_valid,2]
        if compact == True:
            draw_psf_compact(psfs, log_scale=log_scale, fig_name=fig_name)
        else:
            draw_psf_with_center(psfs, ptc_sensor, self.sensor_res,self.sensor_size, log_scale=log_scale,fig_name=fig_name,ruler_len=ruler_len,text_height=text_height)
        
        return psfs,ptc_sensor,pts_stats, valid_index


    def draw_spot_diagram(self, kernel_size = 19, spp = GEO_SPP, grid = 3, fig_name = None, axis="y",scale=1.0,**kwargs):
        """ Draw spot diagram of the lens system.
        
        Args:
            kernel_size (int, optional): size of the kernel for PSF. Defaults to 19.
            spp (int, optional): number of samples per pixel. Defaults to GEO_SPP.
            grid (int, optional): number of points along the grid. Defaults to 3.
            fig_name (str, optional): name of the file to save the spot diagram. Defaults to 'temp/lens_spot_diagram'.
            axis (str, optional): axis to sample the spot diagram. Defaults to "y". alternative is "diag"
        """
        if axis == "y":
            # sample along y-axis
            _,ptc_sensor,pts_stats,_ = self.psf_map(grid=(grid,1), ks=kernel_size, spp=spp, roi_rel=[0,-scale,0,scale],align_corner=True, stats_only=True)
        elif axis == "x":
            # sample along x-axis
            _,ptc_sensor,pts_stats,_ = self.psf_map(grid=(1,grid), ks=kernel_size, spp=spp, roi_rel=[0,0,scale,0],align_corner=True, stats_only=True)
        elif axis == "diag":
            # sample along diagonal
            pts_sensor = self.sample_pts_sensor(grid_size=(grid,grid),roi=[0,0,scale,scale],roi_type="rel", align_corner=True, flip=False) # [grid,grid,2]
            # sample along diagonal
            mask = torch.eye(grid, device=self.device).bool() # [grid,grid]
            pts_sensor = pts_sensor[mask] # [grid,2]
            _,ptc_sensor,pts_stats,_ = self.psf_map(pts_sensor=pts_sensor, ks=kernel_size, spp=spp, align_corner=True, stats_only=True)
        else:
            raise ValueError("axis should be 'y' or 'diag'.")
        pts_shift = pts_stats['raw'] # [spp,grid,2,3]
        ra = pts_stats['ra'] # [spp,grid,3]
        pts_rms = pts_stats['rms'] # [grid]

        print(f"pts RMS (um):",(pts_rms*1000).mean(dim=-1))
        
        
        plot_spot_diagram(pts_shift,ra,ptc_sensor, fig_name = fig_name,**kwargs)
        # plot_spot_diagram(pts_shift,ra,ptc_sensor, fig_name=f'{fig_name}_spot_diagram_x0.5',scale=0.5)

    # ====================================================================================
    # Manipulate lens
    # ==================================================================================== 
    @torch.no_grad()
    def refocus(self, foclen=None):
        """ Refocus the lens to a depth distance by **changing** sensor position.

            Here we simplify the problem by calculating the in-focus position of green light.
        """
        # Trace green light
        _,f_dict = self.calc_foclen(depth="inf")
        z_focus = f_dict['z_focus']
        z_principal = f_dict['z_principal']
        
        if foclen is not None:
            # If focal length is given, calculate the focus position
            z_focus = z_principal + foclen
        
        print(f"Refocus to {z_focus-z_principal}mm, z_focus={z_focus:.2f}mm")
        
        # Update sensor position
        with torch.no_grad():
            self.d_sensor.data = torch.full_like(self.d_sensor,z_focus)
    
    
    # ====================================================================================
    # Loss function
    # ====================================================================================
    def loss_infocus(self, bound=0.005):
        """ Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            bound (float, optional): bound of RMS loss. Defaults to 0.005 [mm].
        """
        loss = []
        for wv in WAVE_RGB:
            # Ray tracing
            ray = self.sample_ray_parallel(fov=0.0, M=31, wvln=wv, entrance_pupil=True)
            ray = self.trace2sensor(ray)

            # Calculate RMS spot size as loss function
            rms_size = torch.sqrt(torch.sum(ray.o**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra))
            loss.append(max(rms_size, bound))
        
        loss_avg = sum(loss) / len(loss)
        return loss_avg


    def loss_surface(self, grad_bound=0.5):
        """ Surface should be smooth, aggressive shape change should be pealized. 
        """
        loss = 0.
        for i in self.find_diff_surf():
            r = self.surfaces[i].r
            loss += max(self.surfaces[i]._dgd(r**2).abs(), grad_bound)

        return loss

    def loss_gap(self, idx, gap_range=10):
        """ Loss function to avoid too small or too large air gaps between lens elements.
        
        Args:
            idx (int): index of the surface to calculate the gap loss.
            gap_range (int, optional): [min_gap, max_gap] for air gaps. Defaults to 10. in [mm].
        """
        gap_min, gap_max = gap_range
        
        loss = torch.tensor(0.0, device=self.device)
        if idx < len(self.surfaces) - 1:
            current_surf = self.surfaces[idx]
            next_surf = self.surfaces[idx+1]

            r = torch.linspace(0.0, 1.0, 20).to(self.device) * current_surf.r
            z_front = current_surf.sag(r, 0) + current_surf.d
            z_next = next_surf.sag(r, 0) + next_surf.d
            
            dist_min = torch.min(z_next - z_front)
            dist_max = torch.max(z_next - z_front)

            loss += max(0, gap_min - dist_min)
            loss += max(0, dist_max - gap_max)
        
        return loss
    
    def loss_self_intersec(self,  thick_range=[0.5,10], air_range=[0.5,100]):
        """ Loss function to avoid self-intersection. Loss is designed by the distance to the next surfaces.
        
        Args:
            thick_range (list, optional): [min_thickness, max_thickness] for lens elements. Defaults to [0.5, 10]. in [mm].
            air_range (list, optional): [min_thickness, max_thickness] for air gaps. Defaults to [0.5, 100]. in [mm].

        Parameter settings:
            General: thick_min=thick_min=0.5 * (total_thickness_of_lens / lens_element_number / 2)
        """
        thick_min, thick_max = thick_range
        air_min, air_max = air_range
        
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            surf_r = min(current_surf.r, self.surfaces[i+1].r)  # use the smaller radius of curvature
            next_surf = self.surfaces[i+1]

            r = torch.linspace(0.0, 1.0, 20).to(self.device) * surf_r
            z_front = current_surf.sag(r, 0) + current_surf.d
            z_next = next_surf.sag(r, 0) + next_surf.d
            
            dist_min = torch.min(z_next - z_front)
            dist_max = torch.max(z_next - z_front)

            if current_surf.mat2.name == 'air':
                # the gap is the air gap
                loss +=  max(0, air_min - dist_min)
                loss +=  max(0, dist_max - air_max)
            else:
                # the gap is the lens thickness
                loss +=  max(0, thick_min - dist_min)
                loss +=  max(0, dist_max - thick_max) 

        return loss


    def loss_last_surf(self, dist_bound=0.6):
        """ The last surface should not hit the sensor plane.

            There should also be space for IR filter.
        """
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.6, 1, 11).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface(r, 0) - last_surf.d
        loss = min(dist_bound, torch.min(z_last_surf))
        return - loss



    def loss_reg(self):
        """ An empirical regularization loss for lens design.
        """
        # For spherical lens design
        loss_reg = 0.1 * self.loss_infocus() + self.loss_self_intersec(dist_bound=0.5, thickness_bound=0.5)
        
        # For cellphone lens design, use 0.01 * loss_reg
        # loss_reg = 0.1 * self.loss_infocus() + self.loss_ray_angle() + (self.loss_self_intersec() + self.loss_last_surf()) #+ self.loss_surface()
        
        
        return loss_reg
    


    # ====================================================================================
    # Optimization
    # ====================================================================================

    def get_optimizer(self, lr, decay=0.2):
        """ Get optimizers and schedulers for different lens parameters.
            For cellphone lens: {"c":1e-4, "d":1e-4, "k":1e-4, "a": 1e-4}
            For camera lens: {"c":1e-3, "d":1e-4, "k":0, "a": 0}
            For DPP lens: {"c":1e-4, "d":1e-4, "zern":1e-4}

        Args:
            lr (dict,optional): learning rate for different parameters.
            epochs (int, optional):  Defaults to 100.
            ai_decay (float, optional): Defaults to 0.2.
        """
        params = self.get_parameters(lr, decay)
        optimizer = torch.optim.Adam(params)
        return optimizer
    
    def get_parameters(self, lr, decay=0.2):
        """ Get optimizers and schedulers for different lens parameters.
            For cellphone lens: {"c":1e-4, "d":1e-4, "k":1e-4, "a": 1e-4}
            For camera lens: {"c":1e-3, "d":1e-4, "k":0, "a": 0}
            For DPP lens: {"c":1e-4, "d":1e-4, "zern":1e-4}

        Args:
            lr (dict,optional): learning rate for different parameters.
            epochs (int, optional):  Defaults to 100.
            ai_decay (float, optional): Defaults to 0.2.
        """
        params = []
        # convert learning rate to float
        for lr_k, lr_v in lr.items(): 
            lr[lr_k] = float(lr_v)
        
        # parameters for Lens Surfaces
        for i in range(len(self.surfaces)):
            surf = self.surfaces[i]
            
            if isinstance(surf, Aspheric):

                params += surf.get_optimizer_params(lr=lr, decay=decay)

            else:
                # print(f'Surface type {type(surf)} not supported yet.')
                params += surf.get_optimizer_params(lr=lr)
                # raise Exception(f'Surface type {type(surf)} not supported yet.')
        
        # parameters for Object Plane
        lr_obj = lr.copy()
        if "d_obj" in lr.keys():
            lr_obj["d"] = lr_obj.pop("d_obj")
        params += self.obj_plane.get_optimizer_params(lr_obj)
        # parameters for Sensor Plane
        if "d_sensor" in lr.keys():
            self.d_sensor.requires_grad = True
            params.append({'params': [self.d_sensor], 'lr': lr["d_sensor"],'name':'d_sensor'})
        # parameters for Color Response Function (CRF)
        if "CRF" in lr.keys():
            params += self.CRF.get_optimizer_params(lr)
        return params

    
    # ====================================================================================
    # Lesn file IO
    # ====================================================================================
    def scale_lens(self, scale):
        """ Scale the lens by a factor.
        
        Args:
            scale (float): scale factor.
        """
        # assume the first surface is at the origin, otherwise, adjust the first surface accordingly
        if self.surfaces[0].d != 0:
            d_shift = self.surfaces[0].d.data
            for i, surface in enumerate(self.surfaces):
                surface.d.data -= d_shift
            self.d_sensor.data -= d_shift
            self.obj_plane.d.data -= d_shift
            
        for surface in self.surfaces:
            surface.scale(scale)
            
        self.sensor_size = [s * scale for s in self.sensor_size]
        self.r_last *= scale
        self.d_sensor *= scale
        self.prepare_sensor(self.sensor_res, sensor_size=self.sensor_size)
        
    def write_lens_json(self, filename='./test.json'):
        """ Write the lens into .json file.
        """
        data = {}
        data['foclen'],_ = self.calc_foclen()
        data['fnum'] = self.calc_fnum(data['foclen'])
        data['r_last'] = self.r_last
        data['d_sensor'] = self.d_sensor.detach().cpu().numpy().tolist()
        data['sensor_size'] = self.sensor_size
        data['CRF'] = {'min':self.CRF.min.detach().cpu().squeeze().tolist(),'max':self.CRF.max.detach().cpu().squeeze().tolist()}
        data['obj_plane'] = self.obj_plane.surf_dict() # save image plane information
        data['surfaces'] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = s.surf_dict()
            if i < len(self.surfaces) - 1:
                surf_dict['d_next'] = self.surfaces[i+1].d.item() - self.surfaces[i].d.item()
            else:
                surf_dict['d_next'] = self.d_sensor.item() - self.surfaces[i].d.item()
            
            data['surfaces'].append(surf_dict)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def get_surface_type(self, surf_type):
        """ Get surface type by its name.
        """
        try:
            return globals().get(surf_type)
        except:
            raise Exception('Surface type not implemented.')
    
            
    def read_lens_json(self, filename='./test.json',device=DEVICE):
        """ Read the lens from .json file.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, 'r') as f:
            data = json.load(f)
            d = 0.0
            self.d_next_list = []
            for surf_idx, surf_dict in enumerate(data['surfaces']):
                if surf_idx == 0 and 'd' in surf_dict.keys():
                    d = surf_dict['d']
                surf_type = surf_dict.pop('type')
                d_next = surf_dict.pop('d_next')
                surface = self.get_surface_type(surf_type)
                if "d" in surf_dict.keys():
                    surf_dict.pop('d') # remove d from the surface dict, as it is not needed here
                s = surface(d=d,device=device,**surf_dict)
                self.surfaces.append(s)
                self.materials.append(Material(surf_dict['mat1']))

                d += d_next
                self.d_next_list.append(d_next)
            self.d_sensor =  torch.tensor(d).to(self.device) # load the last position as the sensor position

        self.materials.append(Material(surf_dict['mat2']))
        self.sensor_size = data['sensor_size']
        
        # find indices of special surfaces
        self.dpp_idx = None
        self.ftl_idx = None
        self.aper_idx = None
        for i, s in enumerate(self.surfaces):
            if isinstance(s, DPP):
                self.dpp_idx = i
            if isinstance(s, FTL):
                self.ftl_idx = i
            if isinstance(s, Aperture):
                if self.aper_idx is not None: # if more than one aperture found, use the last one
                    print(f"Warning: More than one aperture found in the lens, using the last one: {self.aper_idx}.")
                self.aper_idx = i
        
        # configure Object Plane
        if 'obj_plane' in data.keys():
            surface_type = data['obj_plane'].pop('type')
            surface = self.get_surface_type(surface_type)
            self.obj_plane = surface(device=device,**data['obj_plane'])
            print(f'Object plane loaded at distance {self.obj_plane.d} and normal {self.obj_plane.n}.')
        else:
            raise Exception('Object plane not found in the lens file.')
        
        # configure Color Response Function (CRF)
        if 'CRF' in data.keys():
            self.CRF = CRF(data['CRF']['min'],data['CRF']['max']).to(self.device)
        else:
            self.CRF = CRF([0,0,0],[1,1,1]).to(self.device)

