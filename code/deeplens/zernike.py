""" Zernike polynomials and utilities.
Codes adapted from https://github.com/jacopoantonello/zernike.git with torch conversion
"""


from math import factorial
import numpy as np
import torch
from .basics import EPSILON
import torch.nn as nn
from .utils import plot_scatter

def polyval(
        coeffs:torch.tensor, # shape (N,)
        x:torch.tensor, # shape (K,)
        ):
    """ Evaluate a polynomial at a point or array of points.
    The calculation is: 
        y = coeffs[0] * x**(N-1) + coeffs[1] * x**(N-2) + ... + coeffs[N-1] * x**0

    x: torch.tensor, shape (K,)
        The point or array of points at which to evaluate the polynomial.
    coeffs: torch.tensor, shape (N,)
        The coefficients of the polynomial. The order of the coefficients is corresponding to N-1, N-2, ..., 0.
    """
    x = x.unsqueeze(-1) # shape (K,1)
    x_powers = torch.pow(x, torch.arange(coeffs.shape[0]-1,-1,-1).float().to(x.device)) # shape (K,N)
    return torch.sum(x_powers * coeffs, dim=-1) # shape (K,)

def convert_lens2dpp(
        points:torch.tensor,
        facing:int
    ):
    '''
    Convert the coordinate system, because DPP has a different coordinate system.
    The coordinate system of DPP is left-handed, depending on facing
    
    parameters:
        points: torch.tensor, shape (N,2) in lens cartesian coordinates
        facing: int, -1 or 1, the facing direction of the DPP coordinate system
    '''
    x,y = points[...,0],points[...,1]

    if facing == -1:
        '''
            if facing in negative z direction, the coordinate system is:
            x' = -y
            y' = x
            z' = -z
        '''
        return torch.stack([-y,x],dim=-1)
    elif facing == 1:
        '''
            if facing in positive z direction, the coordinate system is:
            x' = -y
            y' = -x
            z' = z
        '''
        return torch.stack([-y,-x],dim=-1)
    else:
        raise ValueError(f"facing must be -1 or 1, but got {facing}")

def convert_dif_dpp2lens(
        difs:torch.tensor,
        facing:int
    ):
    '''
        convert the differential of dpp coordinate system to lens coordinate system.
    
    parameters:
        difs: torch.tensor, shape (N,2) in DPP cartesian coordinates
        facing: int, -1 or 1, the facing direction of the DPP coordinate system
    '''
    dzp_dxp, dzp_dyp = difs[...,0],difs[...,1]
    if facing == -1:
        '''
        if facing in negative z direction, the diff is:
            dz/dx = dz/dzp * dzp/dyp * dyp/dx = -1 * dzp/dyp * 1 = -dzp/dyp
            dz/dy = dz/dzp * dzp/dxp * dxp/dy = -1 * dzp/dxp * -1 = dzp/dxp
        '''
        dz_dx = -dzp_dyp 
        dz_dy = dzp_dxp 
    elif facing == 1:
        '''
        if facing in positive z direction, the diff is:
            dz/dx = dz/dzp * dzp/dyp * dyp/dx = 1 * dzp/dyp * -1 = -dzp/dyp
            dz/dy = dz/dzp * dzp/dxp * dxp/dy = 1 * dzp/dxp * -1 = -dzp/dxp
        '''
        dz_dx = -dzp_dyp 
        dz_dy = -dzp_dxp
    else:
        raise ValueError(f"facing must be -1 or 1, but got {facing}")
    difs = torch.stack([dz_dx,dz_dy],dim=-1)
    
    return difs

class ZernParameter(nn.Module):
    def __init__(self, n, init_val=None, device='cpu'):
        '''
        initalize a sectioned Zernike parameter for the first 12 Zernike polynomials. The total number of Zernike polynomials is 28.
        Onlyt the first n Zernike polynomials are differentiable (which requires grad)

        Parameters 
        ----------
        n : [int] the radial order that requires grad.
        '''
        super().__init__()
        self.n = n
        self.nk = (n + 1) * (n + 2) // 2
        self.sections = []
        self.ntab = torch.zeros(self.nk, dtype=torch.int).to(device)
        for i in range(self.n+1):
            # for i level of zernike polynomial, add i+1 parameters to the sections
            # level 0 has 1 parameter, level 1 has 2 parameters, level 2 has 3 parameters, ...
            self.sections.append(torch.zeros(i+1).to(device))
            self.ntab[i*(i+1)//2:(i+1)*(i+2)//2] = i
        
        self.torch_dtype = torch.float
    
    
    def requires_grad_(self, requires_grad=True):
        for i,section in enumerate(self.sections):
            if i <= self.n:
                section.requires_grad_(requires_grad)

    def set_val(self, val): 
        '''
        set the values of the Zernike parameters. The length of the val should be self.nk, otherwise, it will be padded with zeros.

        Parameters
        ----------
        val: [torch.tensor] or [List] the values of the Zernike parameters, unit is in mm.
        '''
        if type(val) is not torch.Tensor: # convert to tensor (for val is a list or numpy array)
            val = torch.tensor(val,device=self.sections[0].device)
        if len(val) < self.nk:
            val = torch.cat([val,torch.zeros(self.nk - len(val)).to(val.device)]) # fill the rest with zeros
        elif len(val) > self.nk:
            val = val[:self.nk]
            Warning(f"val has more than {self.nk} elements, only the first {self.nk} will be used.")
        assert len(val) == self.nk
        for i in range(len(self.sections)):
            # update the values of the sections
            self.sections[i].data = val[self.ntab==i]
    
    def zero_piston(self):
        self.sections[0].fill_(0)
    
    def zero_tilt(self):
        with torch.no_grad():
            self.sections[1].fill_(0)
    
    def defocus_only(self):
        with torch.no_grad():
            # zero the terms in the second level
            self.sections[2][0].fill_(0)
            self.sections[2][2].fill_(0)
            for i in range(3,self.n+1):
                self.sections[i].fill_(0)
    
    def clip_(self, min_val, max_val):
        for section in self.sections:
            section.clamp_(min_val, max_val)
        
    def forward(self):
        out = torch.cat(list(self.sections))
        return out

class Zernike():
    
    def __init__(self, n=4, r=5, s=1, device='cpu', index='osa',invalid_dict={(0,0)}):
        """Initialise Zernike polynomials up to radial order `n`.

        Parameters 
        ----------
        n : [int] the radial order `n` is the maximum.
        r : [float] the radius of the clear aperture range.
        s : [float] scaling factor for z value.
        device : [str] the device to run the calculations on.
        index : [str] the index type, either 'noll' or 'osa'.
        invalid_dict: [dict] dictionary of invalid Zernike indices (n,m) and their corresponding values.

        """
        self.shape = None
        
        self.torch_dtype = torch.get_default_dtype()

        nk = (n + 1) * (n + 2) // 2
        self.n = n
        self.nk = nk
        self.device = device
        self.r = r  # radius of the clear aperture range
        self.s = s  # scaling factor for z value

        # coefficients of R_n^m(\rho), see [N1976]_
        self.rhotab = torch.zeros(nk, n + 1).to(self.device)
        
        self.ntab = torch.zeros(nk, dtype=torch.int).to(self.device)
        self.mtab = torch.zeros(nk, dtype=torch.int).to(self.device)
        self.coefnorm = torch.zeros(nk).to(self.device)

        self.rhotab[0, n] = 1.0
        self.coefnorm[0] = 1.0


        self.index = index # index type, either 'noll' or 'osa'
        for ni in range(1, n + 1):
            for mi in range(-ni, ni + 1, 2):
                k = self.nm2index(ni, mi, self.index)
                self._make_rhotab_row(k, ni, abs(mi))
                self.ntab[k], self.mtab[k] = ni, mi
        
        # create the derivative of R_n^m(\rho) with respect to \rho
        self.dif_rotab = torch.zeros_like(self.rhotab)
        multipler = torch.arange(n,0,-1).float().to(self.device)
        self.dif_rotab[:, 1:] = self.rhotab[:, :-1] * multipler

        # create invalid Zernike indices based on the current indexing method
        self.invalid_index = []
        for n,m in invalid_dict:
            k = self.nm2index(n, m, self.index)
            self.invalid_index.append(k)

    def _make_rhotab_row(self, c, n, m):
        """ make the `c`-th row of the `rhotab` matrix.
        """
        # col major, row i, col j
        self.coefnorm[c] = self._ck(n, m)
        for s in range((n - m) // 2 + 1):
            self.rhotab[c, self.n - (n - 2 * s)] = (
                ((-1)**s) * factorial(n - s) /
                (factorial(s) * factorial((n + m) // 2 - s) *
                 factorial((n - m) // 2 - s)))
    
    @staticmethod
    def nm2index(n, m, index):
        """Convert indices `(n, m)` to the Noll's index `k`.

        Note that Noll's index `k` starts from one and Python indexing is zero-based.
        """
        if index == 'noll':
            k = n * (n + 1) // 2 + abs(m)
            if (m <= 0 and n % 4 in (0, 1)) or (m >= 0 and n % 4 in (2, 3)):
                k += 1
            k -= 1 
        elif index == 'osa':
            k = (n * (n + 2) + m ) // 2
        else:
            raise ValueError(f"index must be 'noll' or 'osa', but got {index}")
        return k
    
    @staticmethod
    def index2nm(self, k):
        """Convert Noll's index `k` to the indices `(n, m)`.

        Note that Noll's index `k` starts from one and Python indexing is zero-based.
        """
        n = self.ntab[k]
        m = self.mtab[k]
        return n, m
    
    
    def _ck(self, n, m):
        """Normalisation coefficient for the `k`-th Zernike polynomial. """
        
        if m == 0:
            return np.sqrt(n + 1.0)
        else:
            return np.sqrt(2.0 * (n + 1.0))

    def _Rnm(self, k, rho):
        """Compute the `k`-th radial polynomial :math:`R_n^m(\rho)`. """
        
        return polyval(self.rhotab[k, :], rho)

    def _radial(self, k, rho):
        """Compute the radial function for the `k`-th Zernike polynomial.
        """
        return self.coefnorm[k] * self._Rnm(k, rho)
    
    def _angular(self, j, theta):
        """Compute the angular function for the `k`-th Zernike polynomial.
        """
        m = self.mtab[j]
        if m >= 0:
            return torch.cos(m * theta)
        else:
            return torch.sin(-m * theta)
    
    def _dif_angular(self, j, theta):
        """Compute the differential angular function with respect to theta for the `k`-th Zernike polynomial. """
        m = self.mtab[j]
        if m >= 0:
            return -m * torch.sin(m * theta)
        else:
            return -m * torch.cos(-m * theta)
    
    def _normalize(self, points,cartesian=True):
        """ Normalize the coordinates of the points to the range [0, 1]. and return the rho and theta values.
        """
        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)
        if cartesian:
            x,y = points[...,0] / self.r ,points[...,1] / self.r
            rho = torch.sqrt(x**2 + y**2 + EPSILON) 
            theta = torch.atan2(y, x + EPSILON)
        else:
            rho, theta = points[...,0] / self.r ,points[...,1]
            x = rho * torch.cos(theta)
            y = rho * torch.sin(theta)

        return rho, theta , x, y        

    def _Zk(self, k, points,cartesian=True):
        """Compute the `k`-th Zernike polynomial.

        For real-valued Zernike polynomials, :math:`\mathcal{Z}_n^m(\rho,
        \theta) = c_n^m R_n^{|m|}(\rho) \Theta_n^m(\theta)`.
        """
        rho, theta, x, y = self._normalize(points,cartesian=cartesian)

        return self.s * self._radial(k, rho) * self._angular(k, theta)
    

    def _dif_Zk(self, k, points,cartesian=True):
        """Compute the differential of `k`-th Zernike polynomial. """
        # flatten the points if it is a (H,W,2) array
        original_shape = points.shape
        # assert (N, 2) or (H, W, 2), or(N, H,W,2) array
        if points.ndim == 3 or points.ndim == 4:
            points = points.reshape(-1, 2)
        elif points.ndim != 2:
            raise ValueError(f"points must be a (N, 2), (H,W,2), or (N,H,W,2) array points, but got {points.shape}")
        
        rho, theta, x, y = self._normalize(points,cartesian=cartesian)
        
        # calc dz/dr and dz/dtheta
        dzdr = self.coefnorm[k] * polyval(self.dif_rotab[k, :], rho) * self._angular(k, theta)
        dzdtheta = self._radial(k, rho) * self._dif_angular(k, theta)
        
        if cartesian: 
            # calc dz/dx and dz/dy
            # rho = rho + 1e-10 # avoid zero division
            drdx = x/rho
            drdy = y/rho
            dthetadx = -y/rho**2
            dthetady = x/rho**2
            dzdx = dzdr * drdx + dzdtheta * dthetadx
            dzdy = dzdr * drdy + dzdtheta * dthetady
            out = torch.stack([dzdx / self.r , dzdy / self.r],dim=-1)
        else:
            out = torch.stack([dzdr / self.r, dzdtheta],dim=-1)
        
        out = out * self.s # scaling factor
        # reshape the dzdx and dzdy to the original shape if it is a (H,W,2) array
        out = out.reshape(original_shape)
        return out
    
        
    def calc_bases(self, points,cartesian=True):
        """
        Calculate the Zernike basis functions for sampled points

        points: (N,2) array of N points, or (H,W,2) array of HxW points
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        
        return: 
        bases: (N, self.nk) array of N Zernike basis functions
        """
        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)

        # check if points have 2 elements in the last dimension
        if points.shape[-1] != 2:
            raise ValueError("points must have (x,y) or (rho,theta) values in the last dimension")
        
        # assert (N, 2) or (H, W, 2), or(N, H,W,2) array
        if points.ndim == 3 or points.ndim == 4:
            points = points.reshape(-1, 2)
        elif points.ndim != 2:
            raise ValueError(f"points must be a (N, 2), (H,W,2), or (N,H,W,2) array points, but got {points.shape}")
        
        # assert cartesian or polar coordinates
        bases = torch.zeros((points.shape[0], self.nk), dtype=self.torch_dtype).to(self.device)
        for k in range(self.nk):
            if k not in self.invalid_index:
                bases[:, k] = self._Zk(k, points,cartesian=cartesian)
        return bases
    
    def calc_dif_bases(self, points,cartesian=True):
        """
        Calculate the Zernike basis functions differentials for sampled points

        points: (N,2) array of N points, or (H,W,2) array of HxW points
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        
        return: 
        dif_bases: (N, 2, self.nk) array of N Zernike basis functions differentials
        """
        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)

        # check if points have 2 elements in the last dimension
        if points.shape[-1] != 2:
            raise ValueError("points must have (x,y) or (rho,theta) values in the last dimension")
        
        # assert (N, 2) or (H, W, 2), or(N, H,W,2) array
        if points.ndim == 3 or points.ndim == 4:
            points = points.reshape(-1, 2)
        elif points.ndim != 2:
            raise ValueError(f"points must be a (N, 2), (H,W,2), or (N,H,W,2) array points, but got {points.shape}")

        dif_bases = torch.zeros((points.shape[0],2, self.nk), dtype=self.torch_dtype).to(self.device)
        for k in range(self.nk):
            if k not in self.invalid_index:
                dif_bases[..., k] =  self._dif_Zk(k, points,cartesian=cartesian)
        return dif_bases
    

    def eval_points(self, points, params, cartesian=True,coord='lens',facing=-1):
        """
        Evaluate Zernike polynomial at sampled points

        points: (N, 2) array of N (x, y) points
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        params: (self.nk,) array of Zernike coefficients
        coord: if 'lens', it should be converted the points to DPP coordinate
        facing: the facing direction of the DPP device

        return:
        Phi: (N,1) the Phi value at each sampled points
        """
        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)
        
        ## converting the points to DPP coordinate system
        if coord == 'lens':
            points = convert_lens2dpp(points,facing=facing)

        bases = self.calc_bases(points,cartesian)
        phi = torch.matmul(bases, params.to(self.device))
        phi = phi.reshape_as(points[...,0])

        ## addressing the output's coordinate system if needed
        if coord == 'lens' and facing == -1:
            phi = -phi

        return phi
    
    def eval_grad(self, points, params, cartesian=True,coord='lens',facing=-1):
        """Evaluate Zernike polynomial differentials at sampled points

        Parameters:
        ----------
        points: (N, 2) array of N (x, y) points, or (H,W,2) array of HxW (x, y) points in cartesian coordinates
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        params: (self.nk,) array of Zernike coefficients
        coord: if 'lens', it should be converted the points to DPP coordinate system
        facing: the facing direction of the DPP device

        Returns:
        -------
        diffs: (N,2) array of N (dz/dx, dz/dy) differentials at each sampled points
        """ 

        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)
        
        ## converting the points to DPP coordinate system
        if coord == 'lens':
            points = convert_lens2dpp(points,facing=facing)
        
        dif_bases = self.calc_dif_bases(points,cartesian=cartesian)
        # print(f"dtype of params: {params.dtype}, dtype of dif_bases: {dif_bases.dtype}")
        diffs = torch.matmul(dif_bases, params.to(self.device))
        diffs = diffs.reshape_as(points)
        
        ## addressing the output's coordinate system if needed
        if coord == 'lens':
            diffs = convert_dif_dpp2lens(diffs,facing=facing)

        return diffs

    def plot_zern_sag(self, params, fig_name=None,title=f"DPP OPD (um)",z_device=True,**kwargs):
        x,y,z = self.vis_sag(params)
        # if z_device==True, flip the z values
        if z_device:
            z = -z
        x,y,z = x.cpu().detach().numpy(), y.cpu().detach().numpy(), z.cpu().detach().numpy()

        plot_scatter(x,y,z*1000,title=title,fig_name=fig_name,radius=self.r*1.01,**kwargs) # vmin=-2,vmax=2

    def vis_sag(self,params,coord='lens',facing=-1, N=256):
        """
        Visualize the Zernike surface by sampling points on a grid

        Parameters:
        ----------
        params: (self.nk,) array of Zernike coefficients
        coord: if 'lens', it should be converted the points to DPP coordinate system
        facing: the facing direction of the DPP device

        Returns
        -------
        x,y,phi: (HxW), (HxW), (HxW) the x, y, and sag values at each sampled points
        """
        x = torch.linspace(-self.r, self.r, N).to(self.device)
        y = torch.linspace(-self.r, self.r, N).to(self.device)
        x, y = torch.meshgrid(x, y,indexing='ij')
        points = torch.stack([x, y], dim=-1)

        z = self.eval_points(points,params,cartesian=True,coord=coord,facing=facing)
        rho, theta, _,_ = self._normalize(points,cartesian=True)
        mask=[rho <= 1]
        
        return x[mask].flatten(), y[mask].flatten(), z[mask].flatten()

    def vis_grad(self,params, coord='lens',facing=-1):
        """
        Visualize the Zernike surface gradient by sampling points on a grid

        Parameters:
        ----------
        params: (self.nk,) array of Zernike coefficients
        coord: if 'lens', it should be converted the points to DPP coordinate system
        facing: the facing direction of the DPP device

        Returns
        -------
        x,y,phi: (HxW), (HxW), (HxW) the x, y, and sag values at each sampled points
        """
        x = torch.linspace(-self.r, self.r, 256).to(self.device)
        y = torch.linspace(-self.r, self.r, 256).to(self.device)
        x, y = torch.meshgrid(x, y,indexing='ij')
        points = torch.stack([x, y], dim=-1)
        grad = self.eval_grad(points,params,cartesian=True,coord=coord,facing=facing)
        dzdx,dzdy = grad[...,0],grad[...,1]
        rho, theta, _,_ = self._normalize(points,cartesian=True)
        mask=[rho <= 1]
        
        return x[mask].flatten(), y[mask].flatten(), dzdx[mask].flatten(), dzdy[mask].flatten()
    

    def fit_points(self, phi, points, rcond=None, cartesian=True,coord='lens',facing=-1, solver="lstsq"):
        """
        Fit points using least-squares.

        Parameters
        ----------
        phi: (N,) or or (H,W,) the phi value at each sampled points.
        points: (N, 2) array of N (x, y) points, or (H,W,2) array of HxW (x, y) points in cartesian coordinates
        rcond: rcond supplied to `lstsq`
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        coord: if 'lens', it should be converted the points to DPP coordinate system
        facing: the facing direction of the DPP device
        solver: the solver to use, either "transform" or "lstsq". "transform" is faster but less accurate, "lstsq" is more accurate but slower.

        Returns
        -------
        a, `numpy` vector of Zernike coefficients
        res, see `lstsq`
        rnk, see `lstsq`
        sv, see `lstsq`
        """
        invalid_index = self.invalid_index
        self.invalid_index = []
        if phi.ndim == 2: # (H,W) -> (H*W,)
            phi = phi.reshape(-1)
        if type(phi) is not torch.Tensor:
            phi = torch.tensor(phi, dtype=self.torch_dtype,device=self.device)

        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)

        ## converting the points to DPP coordinate system
        if coord == 'lens':
            points = convert_lens2dpp(points,facing=facing)
            phi = facing * phi # flip the sign of phi if facing in negative z direction

        bases = self.calc_bases(points,cartesian=cartesian)
        assert(phi.shape[0] == bases.shape[0])
        rho,theta,x,y = self._normalize(points,cartesian=cartesian)

        mask = (rho <= 1)
        phi1 = phi[mask]
        bases1 = bases[mask]
        self.invalid_index = invalid_index
        if solver == "transform":
            # use the transform method to solve the least squares problem
            a = (phi.unsqueeze(1)*bases).mean(0)
        else:
            a, res, rnk, sv = torch.linalg.lstsq(bases1, phi1, rcond=rcond)
            print(f"residual: {res}, rank: {rnk}, singular values: {sv}")
        return a
        
    
    def fit_difs(self, dir, points, rcond=None, cartesian=True):
        """
        Fit differentials using least-squares.

        Parameters
        ----------
        dir:  the differential value at each sampled points.
        points: (N, 2) array of N (x, y) points, or (H,W,2) array of HxW (x, y) points in cartesian coordinates
        rcond: rcond supplied to `lstsq`
        cartesian: if True, points are in cartesian coordinates, if False, points are in polar coordinates
        coord: if 'lens', it should be converted the points to DPP coordinate system
        facing: the facing direction of the DPP device

        Returns
        -------
        a, `numpy` vector of Zernike coefficients
        res, see `lstsq`
        rnk, see `lstsq`
        sv, see `lstsq`
        """
        
        raise Warning("This function is not tested yet")

        if type(dir) is not torch.Tensor:
            dir = torch.tensor(dir, dtype=self.torch_dtype,device=self.device)
        dir = dir[...,2:]/dir # dz devide dx,dy to get dz/dx and dz/dy
        dir = dir[...,:2]

        if type(points) is not torch.Tensor:
            points = torch.tensor(points, dtype=self.torch_dtype,device=self.device)

        ## converting the points to DPP coordinate system
        if coord == 'lens':
            points = convert_lens2dpp(points,facing=facing)
            dir = convert_dif_dpp2lens(dir,facing=facing)

        dir = dir.reshape(-1,1) # flatten the dir
        dif_bases = self.calc_dif_bases(points,cartesian=cartesian)
        
        dif_bases = dif_bases.reshape(-1,self.nk)
        assert dir.shape[0] == dif_bases.shape[0]        
        mask = torch.isfinite(dir[:,0])
        dir = dir[mask]
        dif_bases = dif_bases[mask]
        a, res, rnk, sv = torch.linalg.lstsq(dif_bases, dir, rcond=rcond)
        return a, res, rnk, sv
    