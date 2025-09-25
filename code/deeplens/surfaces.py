""" Lens surface class.

Code adapted from DeepLens: https://github.com/singer-yang/DeepLens
"""
import torch
import numpy as np
import torch.nn.functional as nnF
import time

from .basics import *
from .utils import *
from .zernike import *


class RotationMatrix(nn.Module):
    def __init__(self, dim=2, params = None, device=DEVICE):
        super().__init__()
        self.dim = dim
        if params is not None:
            assert len(params) == dim * (dim - 1) // 2, 'The number of parameters should be equal to dim * (dim - 1) // 2.'
            self.params = torch.tensor(params)
        else: # initialize as zero
            self.params = torch.zeros(dim * (dim - 1) // 2)
        self.device = device
        self.params.to(device)

    def forward(self):
        """ return a rotation matrix as the exponential of a skew-symmetric matrix."""
        skew_symmetric = torch.zeros(self.dim, self.dim).to(self.device)
        triu_indices = torch.triu_indices(row=self.dim, col=self.dim, offset=1)
        skew_symmetric[triu_indices[0], triu_indices[1]] = self.params
        skew_symmetric = skew_symmetric - skew_symmetric.t()
        return torch.matrix_exp(skew_symmetric)


class Surface(DeepObj):
    def __init__(self, d, mat1, mat2, dif_able=[], device=DEVICE, n = [0,0,1],r=None, h=None, w=None, **kwargs):
        super(Surface, self).__init__()
        self.d = d if torch.is_tensor(d) else torch.Tensor([d])
        if h is not None:
            assert w is not None, 'Height and width should be both provided.'
            self.h = float(h)
            self.w = float(w)
            self.is_rect = True # rectangle
        elif r is not None:
            self.r = float(r) # r is not differentiable
            self.is_rect = False # circle
        else:
            raise Exception('Surface should be either rectangle or circle.')
        
        self.mat1 = Material(mat1)
        self.mat2 = Material(mat2)        

        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 10e-6 # in [mm], here is 10 [nm] 
        self.NEWTONS_TOLERANCE_LOOSE = 50e-6 # in [mm], here is 50 [nm] 
        self.NEWTONS_STEP_BOUND = 5 # [mm], maximum iteration step in Newton's iteration
        self.APERTURE_SAMPLING = 257
        
        self.dif_able = dif_able # list of differentiable parameters

        self.n = torch.tensor(n).float() # normal vector of the plane, always poinint towards positive z-axis

        self.to(device)

    def get_r_max(self):
        """ Get the maximum radius of the surface, used for pruning. By default this is a large value
        """
        return 1000
        
    # ==============================
    # local coordinate system conversion
    # ==============================

    def get_R(self):
        """ Get the uv coordinates on the plane by project the points u,v vectors.
        """
        normal = nnF.normalize(self.n, p = 2, dim = -1)
        v_axis = torch.tensor([0,1.0,0],device=self.device).float() # initialize v_vector as y-axis
        u_axis = nnF.normalize(torch.cross(v_axis, normal), p = 2, dim = -1) # normalized vector
        v_axis = nnF.normalize(torch.cross(normal, u_axis), p = 2, dim = -1) # normalized vector
        R = torch.stack([u_axis, v_axis, normal], dim=-1)

        return R
    
    def get_T(self):
        """ Get the translation vector of the surface.
        """

        p0 = torch.tensor([0,0, 1],device=self.device).float() * self.d # p0 is the point where the surface intersects with the optical z-axis

        return p0

    def glo2loc(self, ray):
        """ Convert global coordinates to local coordinates.

        from global to local the relations is x' = R^T * (x - T),
        where x is the global coordinate, x' is the local coordinate, R is the rotation matrix, T is the translation vector.
        """
        T = self.get_T()
        # shortcut if normal vector is not differentiable
        if self.n.requires_grad == False:
            ray.o = ray.o - T
            return ray
        else:
            R = self.get_R()
            R_inv = R.T
            # assert (R.inverse() == R.T).any(), 'R_inv is not equal to R.T'
            # causion: mind the float type
            new_o = torch.matmul(R_inv, ray.o.unsqueeze(-1) - T.unsqueeze(-1)).squeeze(-1)
            new_d = torch.matmul(R_inv, ray.d.unsqueeze(-1)).squeeze(-1)
            # print("glo2loc:  ", new_o.shape, new_d.shape)
            ray.o = new_o.reshape(ray.o.shape)
            ray.d = new_d.reshape(ray.d.shape)

        return ray
    
    def loc2glo(self, ray):
        """ Convert local coordinates to global coordinates.

        from local to global the relations is x = R * x' + T, 
        where x is the global coordinate, x' is the local coordinate, R is the rotation matrix, T is the translation vector.
        """
        T = self.get_T()
        # shortcut if normal vector is not differentiable
        if self.n.requires_grad == False:
            ray.o = ray.o + T
            return ray
        else:
            R = self.get_R()
            new_o = torch.matmul(R, ray.o.unsqueeze(-1)).squeeze() + T
            new_d = torch.matmul(R, ray.d.unsqueeze(-1)).squeeze()
            # print("loc2glo:  ", new_o.shape, new_d.shape)
            ray.o = new_o.reshape(ray.o.shape)
            ray.d = new_d.reshape(ray.d.shape)

        return ray

    # ==============================
    # Intersection and Refraction
    # ==============================
    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.
        """
        # Determine ray direction and refractive index
        wvln = ray.wvln
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            n1 = self.mat1.ior(wvln)
            n2 = self.mat2.ior(wvln)
        else:
            n1 = self.mat2.ior(wvln)
            n2 = self.mat1.ior(wvln)

        # convert cooordiantes to the local coordinate system
        ray = self.glo2loc(ray)

        # Intersection
        ray = self._intersect(ray, n1)

        # Refraction
        ray = self._refract(ray, n1 / n2)

        # convert cooordiantes back to the global coordinate system
        ray = self.loc2glo(ray)

        return ray
    
    def _intersect(self, ray, n = 1.0):
        """ Solve ray-surface intersection and update ray data.
        """
        # Solve intersection time t by Newton's method
        t, valid = self._newtons_method(ray)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            assert t[valid].min() < 100, 'Precision problem caused by long propagation distance.'
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def _newtons_method(self, ray):
        """ Solve intersection by Newton's method. This function will only update valid rays.

        Notice now the ray is in local coordinate system, therefore the surface is always intersected at z = 0.
        """
        # 1. inital guess of t
        t0 = (0 - ray.o[...,2]) / ray.d[...,2]   # if the shape of aspheric surface is strange, will hit the back surface region instead 

        # 2. use Newton's method to update t to find the intersection points (non-differentiable)
        with torch.no_grad():
            it = 0
            t = t0  # initial guess of t
            ft = MAXT * torch.ones_like(ray.o[...,2])
            while (torch.abs(ft) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < self.NEWTONS_MAXITER):
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[...,0], new_o[...,1]
                valid = self._valid(new_x, new_y) & (ray.ra>0)
                
                ft = self.sag(new_x, new_y, valid) - new_o[...,2]
                dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

            t1 = t - t0

        # 3. do one more Newton iteration to gain gradients
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[...,0], new_o[...,1]
        valid = self._valid(new_x, new_y) & (ray.ra > 0)
        
        ft = self.sag(new_x, new_y, valid)  - new_o[...,2]
        dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
        dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
        dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
        t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

        # determine valid rays
        with torch.no_grad():
            new_x, new_y = new_o[...,0], new_o[...,1]
            valid = self._valid_within_boundary(new_x, new_y) & (ray.ra > 0)
            ft = self.sag(new_x, new_y, valid) - new_o[...,2]
            valid = valid & (torch.abs(ft.detach()) < self.NEWTONS_TOLERANCE_TIGHT) & (t > 0)   # points valid & points accurate & donot go back
        
        return t, valid


    def _refract(self, ray, eta):
        """ Snell's law (surface normal n defined along the positive z axis)
            https://physics.stackexchange.com/a/436252/104805
            https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

            We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it
            points to left. We use the second link to check.

            veci: incident ray
            vect: refractive ray
            eta: relevant refraction coefficient, eta = eta_i/eta_t
        """
        # Compute normal vectors
        n = self._normal(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            n = - n

        # Compute refraction according to Snell's law
        cosi = torch.sum(ray.d * n, axis=-1)   # n * i

        # TIR
        valid = (eta**2 * (1 - cosi**2) < 1) & (ray.ra > 0)

        sr = torch.sqrt(1 - eta**2 * (1 - cosi.unsqueeze(-1)**2) * valid.unsqueeze(-1))  # square root
        
        # First term: vertical. Second term: parallel. Already normalized if both n and ray.d are normalized. 
        new_d = sr * n + eta * (ray.d - cosi.unsqueeze(-1) * n)
        new_d[~valid] = ray.d[~valid]
        
        # Update valid rays
        ray.d = new_d
        ray.ra = ray.ra * valid

        return ray


    def _normal(self, ray):
        """ Calculate normal vector of the surface at intersection point.
        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        nx, ny, nz = self.dfdxyz(x, y)
        n = torch.stack((nx, ny, nz), axis = -1)
        n = nnF.normalize(n, p = 2, dim = -1)

        return n
    
    # =================================================================================
    # Calculation-related methods
    # =================================================================================
    def sag(self, x, y, valid=None):
        """ Calculate sag (z) of the surface. z = f(x, y)

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        if valid is None:
            valid = self._valid(x, y)
        
        x, y = x * valid, y * valid
        return self.g(x, y)

    def dfdxyz(self, x, y, valid=None):
        """ Compute derivatives of surface function. Surface function: f(x, y, z): z - g(x, y) = 0

            NOTE: this function only works for surfaces which can be written as z = g(x, y). For implicit surfaces, we need to compute derivatives (df/dx, df/dy, df/dz).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            dfdx (tensor): df / dx
            dfdy (tensor): df / dy
            dfdz (tensor): df / dz
        """
        if valid is None:
            valid = self._valid(x, y)
        
        x, y = x * valid, y * valid
        dx, dy = self.dgd(x, y)
        return dx, dy, - torch.ones_like(x)
    
    def g(self, x, y):
        """ Calculate sag (z) of the surface. z = f(x, y)

            NOTE: Valid term is used to avoid NaN when x, y are super large, which happens in spherical and aspherical surfaces. But if you want to calculate r = sqrt(x**2, y**2), this will cause another NaN error when calculating dr/dx = x / sqrt(x**2 + y**2). So be careful for this!!!

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        raise NotImplementedError()
        
    def dgd(self, x, y):
        """ Compute derivatives of sag to x and y. (dgdx, dgdy) =  (g'x, g'y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            
        Return:
            dgdx (tensor): dg / dx
            dgdy (tensor): dg / dy
        """
        raise NotImplementedError()
    
    # def is_valid(self, p):
    #     return (self.sdf_approx(p) < 0.0).bool()

    def _valid_within_boundary(self, x, y):
        """ Valid points within the boundary of the surface.
        """
        if self.is_rect:
            valid = self._valid(x, y) & (torch.abs(x) <= self.w/2) & (torch.abs(y) <= self.h/2)
        else:
            valid = self._valid(x, y) & ((x**2 + y**2) <= (self.r)**2)
        
        return valid
    
    def _valid(self, x, y):
        """ Valid points NOT considering the boundary of the surface.
        """
        return torch.ones_like(x, dtype=torch.bool)
    
    def surface_with_offset(self, x, y):
        """ Calculate z coordinate of the surface at (x, y) with offset.
            
            This function is used in lens setup plotting.
        """
        # shortcut if normal vector is [0,0,1]
        
        x = torch.tensor([x]).to(self.device) if type(x) is float else x
        y = torch.tensor([y]).to(self.device) if type(y) is float else y

        if torch.all(self.n == torch.tensor([0,0,1],device=self.device).float()):
            z = self.sag(x, y) + self.d
        else:
            z = torch.ones_like(x) * self.d - 10 * self.c.sign()  # statrting point
            o = torch.stack([x, y, z], dim=-1)
            
            d = torch.zeros_like(o) 
            d[...,2] = self.c.sign() # facing forward
            ray = Ray(o, d, DEFAULT_WAVE, device=self.device)

            local_ray = self.glo2loc(ray)
            ray = self._intersect(local_ray)
            ray = self.loc2glo(ray)

            valid = ray.ra > 0 

            z = ray.o[...,2:]
            if sum(valid) > 0:
                z[z<z[valid].min()] = z[valid].min()
                z[z>z[valid].max()] = z[valid].max()
            else:
                z = torch.ones_like(x) * self.d

        return z.reshape(x.shape)
    
    def max_height(self):
        """ Maximum valid height.
        """
        raise NotImplementedError()
    
    # =========================================
    # Optimization-related methods
    # =========================================
    # def activate_grad(self, activate=True):
    #     raise NotImplementedError()
    
    def get_optimizer_params(self, lr):
        params = []
        type_name = self.__class__.__name__
        if "d" in self.dif_able and "d" in lr.keys():
            self.d.requires_grad_(True)
            params.append({'params': [self.d], 'lr': lr["d"],'name':f'{type_name}_d'})
        if "n" in self.dif_able and "normal" in lr.keys():
            self.n.requires_grad_(True)
            params.append({'params': [self.n], 'lr': lr["normal"],'name':f'{type_name}_n'})
        return params
        # raise NotImplementedError()

    def get_optimizer(self, lr):
        params = self.get_optimizer_params(lr)
        return torch.optim.Adam(params)

    def surf_dict(self):
        normal = nnF.normalize(self.n, p = 2, dim = -1)
        surf_dict = {
            'type': self.__class__.__name__,
            'd': self.d.item(),
            'n': normal.tolist(),
            'mat1': self.mat1.name,
            'mat2': self.mat2.name,
            'dif_able': self.dif_able,
        }
        if self.is_rect:
            surf_dict['h'] = self.h
            surf_dict['w'] = self.w
        else:
            surf_dict['r'] = self.r
        
        return surf_dict
    
    def update(self):
        pass
    
    def disconnect(self):
        pass

    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        raise NotImplementedError()
    
    @torch.no_grad()
    def scale(self, scale):
        self.d = self.d * scale
        if self.is_rect:
            self.h = self.h * scale
            self.w = self.w * scale
        else:
            self.r = self.r * scale
    
    @torch.no_grad()
    def perturb(self, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors.
        """
        raise Exception('This function needs to be implemented in the child class.')
        self.r += np.random.randn() * diameter_precision
        self.d += torch.randn() * thickness_precision


class Aperture(Surface):
    def __init__(self, d, mat1= "air", mat2="air", diffraction=False, device=DEVICE, **kwargs):
        """ Aperture, can be circle or rectangle. 
            For geo optics, it works as a binary mask.
            For wave optics, it works as a diffractive plane.
        """
        Surface.__init__(self, d, mat1, mat2, device=device, **kwargs)
        self.diffraction = diffraction
        self.to(device)

    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.

            In each step, first get a guess of new o and d, then compute valid and only update valid rays. 
        """
        # -------------------------------------
        # Intersection
        # ------------------------------------- 
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) <= self.r) & (ray.ra > 0)

        # => Update position
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        # => Update phase
        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        # -------------------------------------
        # Diffraction
        # ------------------------------------- 
        if self.diffraction:
            raise Exception('Unimplemented diffraction method.')
            # We can use Huygens-Fresnel principle to determine the diffractive rays, but how to make this process differentiable???? Using Heisenberg uncertainty principle???
            # Ref: Simulating multiple diffraction in imaging systems using a path integration method
            # Conventional method process the aperture by using the exit pupil + free space propagation, if that we donot need this class.

        return ray

    def g(self, x, y):
        """ Calculate sag (z) of the surface. z = f(x, y)

            NOTE: Valid term is used to avoid NaN when x, y are super large, which happens in spherical and aspherical surfaces. But if you want to calculate r = sqrt(x**2, y**2), this will cause another NaN error when calculating dr/dx = x / sqrt(x**2 + y**2). So be careful for this!!!

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        return torch.zeros_like(x, device=self.device)  # aperture is flat
    
    def zmx_str(self, surf_idx, d_next):
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
"""
        return zmx_str

class Aspheric(Surface):
    """ This class can represent plane, spheric and aspheric surfaces.

        Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.

        Three kinds of surfaces:
            1. flat: always use round 
            2. spheric: 
            3. aspheric: 
    """
    def __init__(self, d, c=0., k=0., ai=None, mat1=None, mat2=None, device=DEVICE,**kwargs):
        """ Initialize aspheric surface.

        Args:
            r (float): radius of the surface
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): aspheric coefficients
            mat1 (Material): material of the first medium
            mat2 (Material): material of the second medium
            device (torch.device): device to store the tensor
        """
        Surface.__init__(self, d, mat1, mat2, device=device,**kwargs)
        self.c = torch.Tensor([c])
        self.k = torch.Tensor([k])
        # if k == 0:
        #     self.init_k(bound=0.0001)  # if k is 0, initialize it to a small value
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
            elif self.ai_degree == 5:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
                self.ai10 = torch.Tensor([ai[4]])
            elif self.ai_degree == 6:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
            else:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
        else:
            # self.ai = None
            self.ai_degree = 0
            self.init_ai(ai_degree=4)  # default aspheric coefficients
            
        
        self.to(device)


    def init(self, ai_degree=6):
        """ Initialize all parameters.
        """
        self.init_c()
        self.init_k()
        self.init_ai(ai_degree=ai_degree)
        self.init_d()


    def init_c(self, c_bound=0.0002):
        """ Initialize lens surface c parameters by small values between [-0.05, 0.05], 
            which means roc should be (-inf, 20) or (20, inf)
        """
        self.c = c_bound * (torch.rand(1) - 0.5).to(self.device)

    def init_ai(self, ai_degree=3, bound=0.0000001):
        """ If ai is None, set to random value.
            For different length, create a new initilized value and set original ai.
        """
        old_ai_degree = self.ai_degree
        self.ai_degree = ai_degree
        if old_ai_degree == 0:
            if ai_degree == 4:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 0 # * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound * 0 #
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0 # * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0 # * 0.01
            elif ai_degree == 5:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
                self.ai10 = (torch.rand(1, device=self.device)-0.5) * bound* 0.001
            elif ai_degree == 6:
                for i in range(1, self.ai_degree+1):
                    exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')
            else:
                raise Exception('Wrong ai degree')
        else:
            for i in range(old_ai_degree + 1, self.ai_degree + 1):
                exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')

    
    def init_k(self, bound=1):
        """ When k is 0, set to a random value.
        """
        if self.k == 0:
            k = torch.rand(1) * bound
            self.k = k.to(self.device) 


    def init_d(self, bound = 0.1):
        return

    
    def g(self, x, y):
        """ Compute surface height.
        """
        r2 = x**2 + y**2
        total_surface = r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON))

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4
            elif self.ai_degree == 5:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5
            elif self.ai_degree == 6:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5 + self.ai12 * r2 ** 6
            elif self.ai_degree == 7:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + self.ai14 * r2) * r2) * r2) * r2) * r2) * r2) * r2
            elif self.ai_degree == 8:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + (self.ai14 + self.ai16 * r2)* r2) * r2) * r2) * r2) * r2) * r2) * r2
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'total_surface += self.ai{2*i} * r2 ** {i}')

        return total_surface


    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON)
        dsdr2 = (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf)**2

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3
            elif self.ai_degree == 5:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4
            elif self.ai_degree == 6:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4 + 6 * self.ai12 * r2 ** 5
            elif self.ai_degree == 8:
                dsdr2 = dsdr2 + self.ai2 + (2 * self.ai4 + (3 * self.ai6 + (4 * self.ai8 + (5 * self.ai10 + (6 * self.ai12 + (7 * self.ai14 + 8 * self.ai16 * r2)* r2)* r2)* r2) * r2) * r2) * r2 
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'dsdr2 += {i} * self.ai{2*i} * r2 ** {i-1}')

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def _valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        if self.k > -1:
            valid = ((x**2 + y**2) < 1 / self.c**2 / (1 + self.k))
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """ Maximum valid height.
        """
        if self.k > -1:
            max_height = torch.sqrt(1 / (self.k + 1) / (self.c**2)).item() - 0.01
        else:
            max_height = 100

        return max_height

    def get_optimizer_params(self, lr={"c":1e-4, "d":1e-4, "k":1e-4, "a": 1e-4}, decay=0.01):
        """ Get optimizer parameters for different parameters.

        Args:
            lr (list, optional): learning rates for different parameters. Defaults to {"c":1e-4, "d":1e-4, "k":1e-4, "a": 1e-4}
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr*1e3, lr]
        type_name = self.__class__.__name__
        params = []
        if "c" in lr and "c" in self.dif_able:
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr["c"],'name':f'{type_name}_c'})
        if "d" in lr and "d" in self.dif_able:
            self.d.requires_grad_(True)
            params.append({'params': [self.d], 'lr': lr["d"],'name':f'{type_name}_d'})
        if "k" in lr and "k" in self.dif_able:
            self.k.requires_grad_(True)
            params.append({'params': [self.k], 'lr': lr["k"],'name':f'{type_name}_k'})
        if "a" in lr and "a" in self.dif_able:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr["a"] / decay,'name':f'{type_name}_ai2'})
                params.append({'params': [self.ai4], 'lr': lr["a"],'name':f'{type_name}_ai4'})
                params.append({'params': [self.ai6], 'lr': lr["a"] * decay,'name':f'{type_name}_ai6'})
                params.append({'params': [self.ai8], 'lr': lr["a"] * decay**2,'name':f'{type_name}_ai8'})
            elif self.ai_degree == 5:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr["a"] / decay})
                params.append({'params': [self.ai4], 'lr': lr["a"]})
                params.append({'params': [self.ai6], 'lr': lr["a"] * decay})
                params.append({'params': [self.ai8], 'lr': lr["a"] * decay**2})
                params.append({'params': [self.ai10], 'lr': lr["a"] * decay**3})
            elif self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr["a"] / decay})
                params.append({'params': [self.ai4], 'lr': lr["a"]})
                params.append({'params': [self.ai6], 'lr': lr["a"] * decay})
                params.append({'params': [self.ai8], 'lr': lr["a"] * decay**2})
                params.append({'params': [self.ai10], 'lr': lr["a"] * decay**3})
                params.append({'params': [self.ai12], 'lr': lr["a"] * decay**4})
            else:
                for i in range(2, self.ai_degree + 1):
                    exec(f'self.ai{2*i}.requires_grad_(True)')
                    exec(f'params.append({{\'params\': [self.ai{2*i}], \'lr\': lr["a"] / decay**{i-1}}})')
        
        return params


    @torch.no_grad()
    def perturb(self, ratio=0.001, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors. This function should only be called in the final image simulation stage. 
        
        Args:
            ratio (float, optional): perturbation ratio. Defaults to 0.001.
            thickness_precision (float, optional): thickness precision. Defaults to 0.0005.
            diameter_precision (float, optional): diameter precision. Defaults to 0.001.
        """
        self.r += np.random.randn() * diameter_precision
        if self.c != 0:
            self.c *= 1 + np.random.randn() * ratio
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision
        if self.k != 0:
            self.k *= 1 + np.random.randn() * ratio
        for i in range(1, self.ai_degree+1):
            exec(f'self.ai{2*i} *= 1 + np.random.randn() * ratio')


    def surf_dict(self):
        """ Return a dict of surface.
        """
        surf_dict = {
            'type': 'Aspheric',
            'r': self.r,
            'c': self.c.item(),
            'roc': 1 / self.c.item(),
            'd': self.d.item(),
            'k': self.k.item(),
            'ai': [],
            'mat1': self.mat1.name,
            'mat2': self.mat2.name,
            'dif_able': self.dif_able,
            }
        for i in range(1, self.ai_degree+1):
            exec(f'surf_dict[\'ai{2*i}\'] = self.ai{2*i}.item()')
            surf_dict['ai'].append(eval(f'self.ai{2*i}.item()'))

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        assert self.c.item() != 0, 'Aperture surface is re-implemented in Aperture class.'
        assert self.ai is not None or self.k != 0, 'Spheric surface is re-implemented in Spheric class.'
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {self.d.item()}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        return zmx_str


class Cubic(Surface):
    """ Cubic surface: z(x,y) = b3 * (x**3 + y**3)

        Actually Cubic phase is a group of surfaces with changing height.

        Can also be written as: f(x, y, z) = 0
    """
    def __init__(self, d, ai, mat1, mat2, device=DEVICE,**kwargs):
        Surface.__init__(self, d, mat1, mat2, device=device,**kwargs) 
        self.ai = torch.Tensor(ai)

        if len(ai) == 1:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b_degree = 1
        elif len(ai) == 2:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b_degree = 2
        elif len(ai) == 3:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b7 = torch.Tensor([ai[2]]).to(device)
            self.b_degree = 3
        else:
            raise Exception('Unsupported cubic degree!!')

        self.rotate_angle = 0.0

    def g(self, x, y):
        """ Compute surface height z(x, y).
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5) + self.b7 * (x**7 + y**7)
        else:
            raise Exception('Unsupported cubic degre!')
        
        if len(z.size()) == 0:
            z = torch.Tensor([z]).to(self.device)

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)
        
        return z

    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise Exception('Unsupported cubic degree!')

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        return sx, sy

    def get_optimizer_params(self, lr={"d":0.001, "b":0.001}):
        """ Return parameters for optimizer.
        """
        params = super().get_optimizer_params(lr)
        
        # if "d" in self.dif_able:
        #     self.d.requires_grad_(True)
        #     params.append({'params': [self.d], 'lr': lr["d"]})
        type_name = self.__class__.__name__
        if "b" in self.dif_able:
            if self.b_degree == 1:
                self.b3.requires_grad_(True)
                params.append({'params': [self.b3], 'lr': lr["b"],'name':f'{type_name}_b3'})
            elif self.b_degree == 2:
                self.b3.requires_grad_(True)
                self.b5.requires_grad_(True)
                params.append({'params': [self.b3], 'lr': lr["b"],'name':f'{type_name}_b3'})
                params.append({'params': [self.b5], 'lr': lr["b"] * 0.1,'name':f'{type_name}_b5'})
            elif self.b_degree == 3:
                self.b3.requires_grad_(True)
                self.b5.requires_grad_(True)
                self.b7.requires_grad_(True)
                params.append({'params': [self.b3], 'lr': lr["b"],'name':f'{type_name}_b3'})
                params.append({'params': [self.b5], 'lr': lr["b"] * 0.1,'name':f'{type_name}_b5'})
                params.append({'params': [self.b7], 'lr': lr["b"] * 0.01,'name':f'{type_name}_b7'})
            else:
                raise Exception('Unsupported cubic degree!')
        
        return params

    def perturb(self, curvature_precision=0.001, thickness_precision=0.0005, diameter_precision=0.01, angle=0.01):
        """ Perturb the surface
        """
        self.r += np.random.randn() * diameter_precision
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision

        if self.b_degree == 1:
            self.b3 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 2:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 3:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
            self.b7 *= 1 + np.random.randn() * curvature_precision

        self.rotate_angle = np.random.randn() * angle

    def surf_dict(self):
        """ Return surface parameters.
        """

        surf_dict = super().surf_dict()
        surf_dict.update({
            'b3': self.b3.item(),
            'b5': self.b5.item(),
            'b7': self.b7.item(),
            'rotate_angle': self.rotate_angle,
        })
        return surf_dict

class DepthPlane(Surface):
    def __init__(self, d,  mat1, mat2, disp_map, d_range=[-300,0], device=DEVICE, **kwargs):
        """ Plane surface equipped with a depth map, typically rectangle. Working as the object.
            under planar surface assumption, it is aligned without roation.
        
        Args:
            disp_map (tensor): file name for disp map of the surface, relative to d.
            d_range (list): range of the depth map, [min, max]
            d: surface location.
        """
        Surface.__init__(self, d, mat1=mat1, mat2=mat2, device=device, **kwargs)
        self.disp_map = disp_map
        self.d_range = d_range

        """ Load depth map from file. and configure the depth map, together with the pixel size, H,W ."""
        rel_disp = Image.open(disp_map).convert('L')
        rel_disp = transforms.ToTensor()(rel_disp).to(device)[0] # shape: (H, W) ,perform normalization
        
        rel_depth = 1/(rel_disp+0.2)
        self.depth_map = (rel_depth - rel_depth.min())/(rel_depth.max() - rel_depth.min()) # normalize depth
        self.depth_map = d_range[1] + self.depth_map * (d_range[0]-d_range[1]) # scale depth

        print(f"Depth map shape: {self.depth_map.shape}, depth_map_max: {self.depth_map.max()}, depth_map_min: {self.depth_map.min()}")

        self.H, self.W = self.depth_map.shape
        self.pixel_size = self.w / self.W
        assert (self.h / self.H - self.w / self.W)<EPSILON, 'Pixel size should be the same in x and y direction.'
        
        # # Newton's method parameters
        # self.NEWTONS_TOLERANCE_LOOSE = 5e-2
        # self.NEWTONS_TOLERANCE_TIGHT = 1
        # self.NEWTONS_STEP_BOUND = self.d_range[1] - self.d_range[0]
        # self.NEWTONS_MAXITER = 10
        

    def loc2uv(self, ray_o):
        """ Convert local coordinate to uv coordinate.

        Args:
            ray_o (tensor): local coordinate of the ray, should be already on the plane, i.e. z = 0
        """
        return ray_o[...,:2]

    def g(self, x, y):
        """ Compute surface height.
        """
        u = torch.clamp(self.W/2 + x/self.pixel_size, min=-0.99, max=self.W-1.01)
        v = torch.clamp(self.H/2 - y/self.pixel_size, min=-0.99, max=self.H-1.01) 

        # TODO: bilinear interpolation
        idx_i = v.floor().long() + 1
        idx_j = u.floor().long() + 1

        return self.depth_map[idx_i, idx_j]

    def _intersect(self, ray, n=1.0):
        """ Solve ray-surface intersection and update ray data.

        Notice that the surface is always at z = 0 in local corrdinate, so the intersection is simple.
        """
        # Solve intersection
        t = (0 - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d

        # run second time to refine the intersection
        uv = self.loc2uv(new_o[...,:2])
        u, v = uv[...,0], uv[...,1]
        depth = self.g(u, v)
        t = (depth - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = self._valid_within_boundary(u,v) & (ray.ra > 0)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def dgd(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def surf_dict(self):
        """ Return surface parameters.
        """

        surf_dict = super().surf_dict()
        surf_dict.update({
            'disp_map': self.disp_map,
            'd_range': self.d_range,
        })
        return surf_dict

class Plane(Surface):
    def __init__(self, d,  mat1, mat2, device=DEVICE, center=[0,0],R_img = [0.0], **kwargs):
        """ Plane surface, typically rectangle. Working as IR filter, lens cover or DOE base.
        """
        Surface.__init__(self, d, mat1=mat1, mat2=mat2, device=device, **kwargs)
        self.center = torch.tensor(center, device=device).float() # center is the relative local center coordinate of the plane
        self.R_img = RotationMatrix(dim=2,params=R_img,device=device)

    def _intersect(self, ray, n=1.0):
        """ Solve ray-surface intersection and update ray data.

        Notice that the surface is always at z = 0 in local corrdinate, so the intersection is simple.
        """
        # Solve intersection
        t = (0 - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        
        # rot = self.R_img() # return a 2x2 rotation matrix
        # assert abs(torch.det(rot).item()-1) < 1e-4, 'The determinant of rotation matrix should be 1'
        # new_o[...,:2] = torch.matmul(rot,(new_o[...,:2]-self.center).float().unsqueeze(-1)).squeeze(-1) # apply translation and rotation in 2D plane
        # u,v = new_o[...,0], new_o[...,1]

        # assert validity with center and rotation check
        uv = self.loc2uv(new_o[...,:2])
        u, v = uv[...,0], uv[...,1]
        valid = self._valid_within_boundary(u, v) & (ray.ra > 0)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def loc2uv(self, ray_o):
        """ Convert local coordinate to uv coordinate.

        Args:
            ray_o (tensor): local coordinate of the ray, should be already on the plane, i.e. z = 0
        """
        rot = self.R_img() # return a 2x2 rotation matrix
        assert abs(torch.det(rot).item()-1) < 1e-4, 'The determinant of rotation matrix should be 1'
        uv = torch.matmul(rot,(ray_o[...,:2]-self.center).unsqueeze(-1)).squeeze(-1) # apply translation and rotation in 2D plane
        # uv = torch.matmul(rot,(ray_o[...,:2]-self.center).float().unsqueeze(-1)).squeeze(-1) # apply translation and rotation in 2D plane

        return uv[...,:2]


    
    def g(self, x, y):
        return torch.zeros_like(x)
    
    def dgd(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def get_optimizer_params(self, lr):
        params = super().get_optimizer_params(lr)
        type_name = self.__class__.__name__
        if "center" in self.dif_able and "center" in lr.keys():
            self.center.requires_grad_(True)
            params.append({'params': [self.center], 'lr': lr["center"],'name':f'{type_name}_center'})
        if "R_img" in self.dif_able and "R_img" in lr.keys():
            self.R_img.params.requires_grad_(True)
            params.append({'params': [self.R_img.params], 'lr': lr["R_img"],'name':f'{type_name}_R_img'})
        return params

    def surf_dict(self):
        surf_dict = super().surf_dict()
        surf_dict['center'] = self.center.tolist()
        surf_dict['R_img'] = self.R_img.params.tolist()
        return surf_dict

    def surface_sample(self, N=1000):
        """ Sample uniform points on the surface location d.
        
        Note: this funcion is only used in optics.py for function: refocus(), and calc_foc_dist().
                It is noly an estimation and should be deprecated in the future.
        """
        r_max = self.r
        theta = torch.rand(N)*2*np.pi
        r = torch.sqrt(torch.rand(N)*r_max**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, self.d.item())
        o2 = torch.stack((x2,y2,z2), 1).to(self.device)
        return o2

class DPP(Surface):
    def __init__(self,r, d, mat1, mat2, ref_idx, zern_order=4, device=DEVICE, zern_amp=None, param_range=None, apply_hardware=False, update_tilt=False, defocus_only=False, **kwargs):
        """ Deformable Phase Plate (DPP) surface, typically circle.

        Args:
            zern_amp: [torch.tensor] or [List] the values of the Zernike parameters, unit is in mm.
        """  
        try: 
            from .drivers import DPP_CTL
        except:
            raise Exception('Please install the drivers for DPP control')

        Surface.__init__(self, d, mat1=mat1, mat2=mat2, r=r, device=device,**kwargs)
        self.zern = Zernike(n=zern_order, device=device, r=r, s=1/(ref_idx-1))

        if param_range is None: # default range for zernike [-0.004,0.004] mm
            param_range = [-0.004,0.004]
        else:
            assert len(param_range) == 2, 'param_range should be a list with two elements'
        self.param_range = param_range 

        self.hardware = DPP_CTL(device=device,connect=apply_hardware)
        
        self.apply_hardware = apply_hardware
        self.update_tilt = update_tilt
        self.defocus_only = defocus_only


        self.zern_sec = ZernParameter(zern_order,device=self.device)
        if zern_amp is not None:
            if self.zern.nk != len(zern_amp):
                print('zern_amp should have the same length as zern.nk, now converting it by assuming the rest is zero') 
            # self.zern_amp = torch.tensor(zern_amp,device=self.device).float()
            self.zern_sec.set_val(zern_amp)

        if not self.update_tilt:
            self.zern_sec.zero_tilt()
        if self.defocus_only:
            self.zern_sec.defocus_only()

        self.update(clip=False)

    def reset_zern_order(self, zern_order):
        """ Reset the Zernike order of the DPP surface.
        """
        self.zern_sec = ZernParameter(zern_order, device=self.device)
        self.zern = Zernike(n=zern_order, device=self.device, r=self.r, s=self.zern.s)

    def set_zern_amp(self, zern_amp):
        """ Set the zern_amp of self DPP surface.
        """
        with torch.no_grad():
            self.zern_sec.set_val(zern_amp)
        self.update(clip=False)

    def get_zern_amp(self,truncated=False):
        """
            This function is called every time when zern_amp is needed, it will return the zern_amp from different control parameters.
        
        Params:
            truncated: [bool] if True, the zernike amplitude will be truncated to first nk terms
        Return:
            zern_amp: [torch.Tensor] of shape(91). The zernike amplitude in the range of param_range, in (mm)
        """
        zern_amp = self.zern_sec()
        if truncated:
            zern_amp = zern_amp[:self.zern.nk]
        return zern_amp
 
    def get_volt(self):
        """
            This function is called every time when volt is needed, it will return the volt from different control parameters.

        Return:
            volt: torch.Tensor, the voltage in the range of param_range, in (V)
        """
        volt = self.hardware.cal_volt_from_zern(self.zern_sec())
        return volt

    def update(self,clip=True):
        """ Update zern parameters to be in the range of zern range, this operation replace the original zern parameters."""
        vmin,vmax = self.param_range

        with torch.no_grad():
            # clipping, zero_piston and zero_tilt are already included in the zern_amp_section
            self.zern_sec.clip_(vmin,vmax) # clip the zern_amp to the range
            if clip:
                self.zern_sec.zero_piston() # enforce the first element (piston) to be zero
                if not self.update_tilt:
                    self.zern_sec.zero_tilt() # disabled the tilt if needed
                if self.defocus_only:
                    self.zern_sec.defocus_only() # only keep the defocus term

            if self.apply_hardware:
                volt = self.get_volt()
                print(f'Applying voltage: {volt}')
                ## unified all the hardware control by voltage
                volt_written,converted_voltages = self.hardware.apply_volt(volt.detach().cpu().numpy())
                assert volt_written == True, 'Failed to write voltage to hardware'
                time.sleep(0.3)
            
        
    def get_optimizer_params(self, lr={"zern":0.001, "d":0.001}):
        """ Activate gradient computation for c and d and return optimizer parameters.
        """
        params = super().get_optimizer_params(lr)
        type_name = self.__class__.__name__
        if "zern" in self.dif_able and "zern" in lr.keys():
            ## set different learning rate for different zernike terms
            self.zern_sec.requires_grad_(True)
            for i in range(len(self.zern_sec.sections)):
                if i <= self.zern_sec.n:
                    # different learning rate for different zernike terms, the learning rate is divided by 10 for each 4th order
                    params.append({'params': [self.zern_sec.sections[i]], 'lr': lr["zern"]*10 / 10**(i/2),'name':f'{type_name}_zern_{i}'}) 
                    # params.append({'params': [self.zern_sec.sections[i]], 'lr': lr["zern"]*10,'name':f'{type_name}_zern_{i}'}) 

        return params

    def _valid(self, x, y):
        """ Invalid when shape is non-defined. 
        """
        valid = (x**2 + y**2 < self.r**2)
        return valid
    
    def disconnect(self):
        self.hardware.disconnect()

    def surf_dict(self):

        surf_dict = super().surf_dict()
        # volt = self.get_volt()
        zern_amp = self.get_zern_amp()

        surf_dict.update({
            'zern_order': self.zern.n,
            'apply_hardware': self.apply_hardware,
            'update_tilt': self.update_tilt,  
            'defocus_only': self.defocus_only,
            'param_range': self.param_range,
            'zern_amp': zern_amp.tolist(),
        })
        return surf_dict


class DPP_DOE(DPP):
    def __init__(self,r, d, mat1, mat2, **kwargs):
        """ Deformable Phase Plate (DPP) surface, typically circle. Simplified version of DPP, only consider the phase shift.

        """
        assert mat2== 'air' and mat1 == 'air', 'simple DPP only works for air to air.'
        # make n=2 to ensure scale of DPP is 1.0
        DPP.__init__(self,r, d, mat1=mat1, mat2=mat2, ref_idx=2, **kwargs)

    def _intersect(self, ray, n = 1.0):
        """ directly return ray
        
        notice: this version assumes self.d = 0 in the local coordinate system
        """
        # ray.propagate_to(self.d)

        # Solve intersection
        t = (0 - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        u,v = new_o[...,0], new_o[...,1]
        valid = (torch.sqrt(u**2 + v**2) <= self.r) & (ray.ra > 0)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t 
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def _refract(self, ray, eta):
        """ simlulate the refraction by adding the phase shift.
        
        The DPP is calibrated using 632nm wavelength, in this simulation the amplitude is in unit of mm of OPD. 
        As the device is refractive in priciple, actually the phase-shift is different, but the OPD is considered constant for all wavelengths.
            ref. 
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] https://support.zemax.com/hc/en-us/articles/1500005491181-How-to-design-DOE-lens-or-metalens-in-OpticStudio

        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        valid = self._valid(x, y)
        x, y = x * valid, y * valid
        points = torch.stack([x, y], dim=-1)
        if points.dim() == 1:
            points = points.unsqueeze(0)

        # Update valid ray direction
        zern_amp = self.get_zern_amp(truncated=True)
        n_xy = self.zern.eval_grad(points,zern_amp,coord='lens') # the gradient of the phase shift, should have dz/dx and dz/dy
        
        # perform the direction update on cosine of x,y direction
        ray_xy = ray.d[...,:2] - n_xy #* ray.wvln / 0.632
        
        # Ensure the z-component makes the vector normalized
        ray_z = torch.sqrt(1 - torch.sum(ray_xy**2,axis=-1))

        # adjust the ray_z to the correct direction
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if not forward:
            ray_z = -ray_z
        # ray.d[...,2] = ray_z # this operation is suspicus to be non-differentiable
        ray_d = torch.cat([ray_xy, ray_z.unsqueeze(-1)],dim=-1)
        ray.d = nnF.normalize(ray_d, p = 2, dim = -1)
        
        # Update valid ray 
        ray.ra = ray.ra * valid
        
        # Update valid ray opl
        if ray.coherent:
            opd = self.zern.eval_points(points,zern_amp,coord='lens')
            if forward:
                opd = -opd
            # print(f'OPD max: {opd.max().item()}, min: {opd.min().item()}')
            ray.opl = ray.opl + opd

        return ray
    
    def g(self, x, y):
        return torch.zeros_like(x)
    

class DPP_physical(DPP):
    def __init__(self, r, d, mat1, mat2, **kwargs):
        """ Deformable Phase Plate (DPP) surface, typically circle.

        """
        if mat2!= 'air':
            ref_idx = Material(mat2).ior(DEFAULT_WAVE)
        else:
            ref_idx = Material(mat1).ior(DEFAULT_WAVE)
        DPP.__init__(self, r, d, mat1=mat1, mat2=mat2, ref_idx=ref_idx,**kwargs)
    
    def g(self, x, y):
        """
        evaluate the sag function of the DPP surface
        """

        # operate on the `prime` coordinate system
        points = torch.stack([x, y], dim=-1)
        if points.dim() == 1:
            points = points.unsqueeze(0)
        zern_amp = self.get_zern_amp(truncated=True)
        z = self.zern.eval_points(points,zern_amp,coord='lens')
        return z
    
    def dgd(self, x, y):
        """
        Evaluate the gradient of the sag function with respect to x and y.
        """

        # operate on the `prime` coordinate system
        points = torch.stack([x, y], dim=-1) # change in coordinate system
        if points.dim() == 1:
            points = points.unsqueeze(0)
        zern_amp = self.get_zern_amp(truncated=True)
        grad = self.zern.eval_grad(points,zern_amp,coord='lens')
        dz_dx, dz_dy = grad[...,0], grad[...,1]

        return dz_dx, dz_dy
    
class Spheric(Surface):
    """ Spheric surface.
    """
    def __init__(self, roc, d, mat1, mat2, device=DEVICE, thick=0,**kwargs):
        super().__init__(d, mat1, mat2, device=device,**kwargs)
        self.c = torch.tensor([1/roc])
        self.hlf_thick = thick/2.0 # half thickness of the lens
        self.to(device)
    
    def get_r_max(self):
        return 1/self.c.abs().item()*0.99

    def get_T(self):
        """ Get the translation vector of the surface.
        """
        p0 = torch.tensor([0,0, 1],device=self.device).float() * self.d ## p0 is the point on vertex point
        
        if self.n.requires_grad:
            # find the tilted origin point
            normal = nnF.normalize(self.n, p = 2, dim = -1)
            p0 = p0 + (torch.tensor([0,0, self.hlf_thick],device=self.device) - normal * self.hlf_thick) * self.c.sign()

        return p0


    def g(self, x, y):
        """ Compute surfaces sag z = (1 - sqrt(1 - r**2 * c**2)) / c
        """

        r2 = x**2 + y**2
        sag = self.c * r2 / (1 + torch.sqrt(1 - r2 * self.c**2))
        # sag = (1 - torch.sqrt(1 - r2 * self.c**2)) / self.c

        return sag

    def dgd(self, x, y):
        """ Compute surface sag derivatives to x and y: dz / dx, dz / dy.
        """

        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * self.c**2)
        # dgdr2 = (1 + sf + r2 * self.c**2 / 2 / sf) * self.c / (1 + sf)**2
        dgdr2 =  self.c / (2*sf)
        # assert (dgdr2 - dgdr2_new).sum() < 1e-4, f'{dgdr2} != {dgdr2_new}'
        dx, dy = dgdr2*2*x, dgdr2*2*y

        return dx, dy

    def _valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        # nx,ny,nz = self.n
        # tilt = -(nx*x + ny*y)/nz
        # p = torch.stack([x, y, tilt], dim=-1)
        # u,v = self.get_uv(p)
        u,v = x, y
        valid = (u**2 + v**2 < 1 / self.c**2 )
        return valid
    
    def max_height(self):
        """ Maximum valid height.
        """
        max_height = torch.sqrt(1 / self.c**2).item() - 0.01
        return max_height
    
    def get_optimizer_params(self, lr={"c":0.001, "d":0.001}):
        """ Activate gradient computation for c and d and return optimizer parameters.
        """
        params = super().get_optimizer_params(lr)
        # if "d" in self.dif_able and "d" in lr.keys():
        #     self.d.requires_grad_(True)
        #     params.append({'params': [self.d], 'lr': lr["d"]})
        # if "n" in self.dif_able and "normal" in lr.keys():
        #     self.n.requires_grad_(True)
        #     params.append({'params': [self.n], 'lr': lr["normal"]})
        type_name = self.__class__.__name__
        if "c" in self.dif_able and "c" in lr.keys():
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr["c"],'name':f'{type_name}_c'})
        return params

    def surf_dict(self):
        surf_dict = super().surf_dict()
        surf_dict.update({
                'roc': 1/self.c.item(),
                'thick':self.hlf_thick*2,
            })

        return surf_dict
    
    @torch.no_grad()
    def scale(self, scale):
        """ Scale the surface by a factor.
        """
        super().scale(scale)
        self.c /= scale
        self.hlf_thick *= scale
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r*2}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r*2}
"""

        return zmx_str


class FTL(Spheric):
    """ Focus Tunable Lense surface.
    """
    def __init__(self, c, c_range, d, mat1="air", mat2="air", device=DEVICE,apply_hardware=False,**kwargs):
        super(FTL, self).__init__(d, mat1, mat2, device=device,**kwargs)

        if mat2!= 'air':
            self.ref_idx = self.mat2.ior(DEFAULT_WAVE)
        else:
            self.ref_idx = self.mat1.ior(DEFAULT_WAVE)

        self.c = torch.tensor([c]).to(device)
        self.c_range = c_range
        # self.to(device)
        # refer to the optotune modeling: https://support.zemax.com/hc/en-us/articles/13019196463123-How-to-use-OPTOTUNE-focus-tunable-liquid-lenses-in-OpticStudio
        # notice here the most significant coma is about 0.5 Wavefront, converting to around 0.00059
        # self.zern = Zernike(3, device=device, r=r, s=1)
        # self.zern_amp = torch.zeros(self.zern.nk).to(self.device).float()
        # self.zern_amp[6] = coma
        self.hardware = None
        self.apply_hardware = apply_hardware
        if self.apply_hardware:
            try: 
                from .drivers import FTL_CTL
            except:
                raise Exception('Please install the drivers for FTL control')
            self.hardware = FTL_CTL()
    
    
    def get_optimizer_params(self, lr={"c":0.001, "d":0.001}):
        """ Activate gradient computation for c and d and return optimizer parameters.
        """
        params = super().get_optimizer_params(lr)
        type_name = self.__class__.__name__
        if "c" in self.dif_able and "c" in lr.keys():
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr["c"],'name':f'{type_name}_c'})
        return params

    def surf_dict(self):
        dpt = self.c.item() * 1000 * (self.ref_idx - 1)
        surf_dict = super().surf_dict()
        surf_dict.update({
                'c_range': self.c_range,
                'dpt': dpt,
                'apply_hardware': self.apply_hardware,
                # 'coma':self.zern_amp[6].item(),
            })

        return surf_dict
    
    def update(self):
        """ Update c to be in the range of c_range """
        with torch.no_grad():
            self.c.clip_(self.c_range[0], self.c_range[1])
        if self.hardware is not None:
            dpt = self.c.item() * 1000 * (self.ref_idx - 1)
            print("Set Focal Power to hardware: ", dpt)
            self.hardware.SetFocalPower(dpt)
            time.sleep(0.1) # sleep for 0.1s to wait for the hardware to response
    
    def disconnect(self):
        if self.hardware is not None:
            self.hardware.disconnect()
