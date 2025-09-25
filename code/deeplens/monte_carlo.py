""" Forward and backward Monte-Carlo integral functions.
"""
import torch
from .basics import EPSILON
from .utils import cvt_img2pix,calc_psf_stats

def forward_integral(ray, ps, ks, pointc_ref=None, stats_only=False):
    """ Forward integral model, including PSF and vignetting

    Args:
        ray: Ray object. Shape of ray.o is [spp, N, 3].
        ps: pixel size
        ks: kernel size.
        pointc_ref: reference pointc, shape [N,2]. If None, use averaged center.
        stats_only: whether to return only the mean square error of the pts

    Returns:
        psf: point spread function, shape [N, ks, ks]
    """
    pts = ray.o[..., :2]       # shape [spp, N, 2] or [spp, 2].
    
    # ==> PSF center
    if pointc_ref is None:
        # Use averaged center
        pointc_ref = (pts * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON) # shape [2]

    pts_shift = pts - pointc_ref.to(pts.device)
    
    N_ra = ray.ra.sum(0) # shape [N] .add(EPSILON)
    assert N_ra.all() > 0, "Some rays are not valid. Please check the ray."
    
    # ==> Calculate PSF statistics
    pts_stats, pts_r = calc_psf_stats(pts_shift, ray.ra)

    if stats_only:
        # save computation if only mse is needed
        spp,N,_ = pts_shift.shape # spp: samples per point
        psf = torch.ones(N, ks, ks).to(pts.device)
        cover_ratio = N_ra / spp
    else:
        # calc valid pts within the psf_range for psf rendering
        ra = ray.ra * (pts_shift[...,0].abs() < (ks / 2) * ps) * (pts_shift[...,1].abs() < (ks / 2) * ps)   # shape [spp, N] or [spp].
        cover_ratio = ra.sum(0) / N_ra # shape [N]

        # ==> Calculate PSF
        obliq = ray.d[..., 2]**2 # notice obliq is not used, but keep it for future use
        psf = assign_pts_to_pixels(pts=pts_shift, ks=ks, ps = ps, ra=ra, obliq=obliq)
    pts_stats['cover_ratio']=cover_ratio

    return psf,pointc_ref, pts_stats


def assign_pts_to_pixels(pts, ks, ps, ra, interpolate=True, coherent=False, phase=None, d=None, obliq=None, wvln=0.589):
    """ Assign pts to pixels, both coherent and incoherent. Use advanced indexing to increment the count for each corresponding pixel. This function can only compute single point source, single wvln. If you want to compute multiple point or muyltiple wvln, please call this function multiple times.
    
    Args:
        pts: shape [spp, N, 2], or [spp, 2]
        ks: kernel size in number of pixels
        ps: pixel size in mm
        ra: shape [spp, N], or [spp,]
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp, N, 1]

    Returns:
        psf: shape [N, ks, ks], or [ks, ks]. PSF is a 2D grid in [H,W] image coordinate system.
    """
    # pts *= ra.unsqueeze(-1)
    # ==> Parameters
    device = pts.device
    pts_pix = cvt_img2pix(pts, (ks, ks), (ps*ks,ps*ks)) # convert pts to pixel coordinates in [0, ks-1] range, x right, y down
    u,v = pts_pix[..., 0], pts_pix[..., 1] # shape [spp, N] or [spp]
    ra = ra * (u >= 0) * (u < ks) * (v >= 0) * (v < ks) # mask out pts outside the range [0, ks)
    u,v = torch.clamp(u-0.5, 0, ks-1.01), torch.clamp(v-0.5, 0, ks-1.01) # shift to center of pixel, and clamp to [0, ks-1-EPSILON] range

    pts_ij = torch.stack((v,u), dim=-1) # shape [spp, N, 2] or [spp, 2]
    

    if pts.dim() == 3:
        spp, N, _ = pts.shape
        seq_index = torch.arange(N,dtype=torch.long).to(device).repeat(spp, 1) # to shape [spp, N]
        seq_index = seq_index.flatten() # to shape [spp*N]
        grid = torch.zeros(N, ks, ks).to(device)

        def to_index(pixel_indices):
            # connect the pixel_index with the seq_index
            # pixel_indices: shape [spp, N, 2]
            pixel_idx = pixel_indices.permute(2,0,1).flatten(1) # to shape [2, spp*N]
            # print(f"shape of pixel_indices: {pixel_indices.shape}")
            # concate the seq_index to the pixel_indices
            out_index = torch.cat((seq_index.unsqueeze(0), pixel_idx), dim=0) # to shape [3, spp*N]
            return tuple(out_index.long())

    elif pts.dim() == 2:
        spp = pts.shape[0]
        grid = torch.zeros(ks, ks).to(device)
        def to_index(pixel_indices):
            return tuple(pixel_indices.t().long())


    if interpolate:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        i,j = pts_ij[..., 0], pts_ij[..., 1] # shape [spp, N] or [spp]
        idx_i, idx_j = i.floor(), j.floor() # shape [spp, N] or [spp]
        assert max(idx_i.max(), idx_j.max()) < ks, f"Pixel indices {idx_i.max()}, {idx_j.max()} exceed kernel size {ks}."
        
        w_b = i - idx_i # weight for bottom, range [0, 1], equals 1 when reach bottom pixel
        w_r = j - idx_j # weight for right, range [0, 1], equals 1 when reach right pixel

        # ==> Pixel indices
        pixel_indices_tl = torch.stack((idx_i, idx_j), dim=-1)
        pixel_indices_tr = torch.stack((idx_i, idx_j+1), dim=-1)
        pixel_indices_bl = torch.stack((idx_i+1, idx_j), dim=-1)
        pixel_indices_br = torch.stack((idx_i+1, idx_j+1), dim=-1)

        if coherent:
            # ==> Use advanced indexing to increment the count for each corresponding pixel
            grid = grid + 0j
            grid.index_put_(to_index(pixel_indices_tl), ((1-w_b)*(1-w_r)*ra*torch.exp(1j*phase)).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_tr), ((1-w_b)*w_r*ra*torch.exp(1j*phase)).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_bl), (w_b*(1-w_r)*ra*torch.exp(1j*phase)).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_br), (w_b*w_r*ra*torch.exp(1j*phase)).flatten(), accumulate=True)

        else:
            grid.index_put_(to_index(pixel_indices_tl), ((1-w_b)*(1-w_r)*ra).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_tr), ((1-w_b)*w_r*ra).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_bl), (w_b*(1-w_r)*ra).flatten(), accumulate=True)
            grid.index_put_(to_index(pixel_indices_br), (w_b*w_r*ra).flatten(), accumulate=True)

    else:
        pixel_indices_tl = pts_ij.floor().long()
        grid.index_put_(to_index(pixel_indices_tl), ra.flatten(), accumulate=True)
        
    return grid

