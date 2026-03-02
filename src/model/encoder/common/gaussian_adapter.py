from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def pts2covariance(points):
    """
    Args:
        torch.Tensor: 
        point position, [b, v, h, w, 3].
    
    Returns:
        torch.Tensor:
        covariances, [b, v, h, w, 3, 3], 
        quats, [b, v, h, w, 4], 
        scales, [b, v, h, w, 3].
    """
    b, v, h, w, c = points.shape
    assert c == 3, "3-d coord."

    sobel_x = torch.tensor([[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]], dtype=torch.float32, device=points.device).view(1, 1, 3, 3).repeat(3,1,1,1)
    sobel_y = torch.tensor([[-1/8, -2/8, -1/8], [0, 0, 0], [1/8, 2/8, 1/8]], dtype=torch.float32, device=points.device).view(1, 1, 3, 3).repeat(3,1,1,1)

    # [n * v, c, h, w]
    points = points.view(b * v, h, w, c).permute(0, 3, 1, 2)  # [(b v), c, h, w]
    depths = points[:, 2, :, :]  # [(b v), h, w]
    
    grad_x = F.conv2d(points, sobel_x, groups=c)  
    grad_y = F.conv2d(points, sobel_y, groups=c)  

    normals = torch.cross(grad_x, grad_y, dim=1) 
    normals = F.pad(normals, (1, 1, 1, 1), mode='replicate')  
    normals = normals.permute(0, 2, 3, 1).view(b, v, h, w, c)  
    normals = F.normalize(normals, dim=-1)

    rotations = build_gs_rotation_from_normal(normals.reshape(b*v*h*w, c))
    rotations = rotations.view(b, v, h, w, c, c)
    
    grad_x = F.pad(grad_x, (1, 1, 1, 1), mode='replicate').clamp(min=-0.1,max=0.1)
    grad_y = F.pad(grad_y, (1, 1, 1, 1), mode='replicate').clamp(min=-0.1,max=0.1)
    
    delta_x_x = grad_x[:, 0, :, :].reshape(-1) # (b v h w)
    delta_x_y = grad_y[:, 0, :, :].reshape(-1)
    delta_y_x = grad_x[:, 1, :, :].reshape(-1)
    delta_y_y = grad_y[:, 1, :, :].reshape(-1)
    delta_z_x = grad_x[:, 2, :, :].reshape(-1)
    delta_z_y = grad_y[:, 2, :, :].reshape(-1)

    scale_x = torch.sqrt(delta_x_x**2 + delta_z_x**2)
    scale_y = torch.sqrt(delta_y_y**2 + delta_z_y**2)  
    scales = torch.stack((scale_x, scale_y), dim=-1)  # [batch, h, w, 2]
    scales = scales.view(b, v, h, w, 2)  # [batch, v, h, w, 2]
    
    return rotations, scales, normals


    
def build_gs_rotation_from_normal(normal):
    """
    input normalized normal
    default normal is (0,0,1)
    """
    start_normal = torch.tensor([0., 0., 1.], device=normal.device, dtype=normal.dtype).repeat(normal.shape[0], 1)
    normal = normal * torch.sign(normal[:, 2:3])
    # Compute cross product and dot product
    v = torch.cross(start_normal, normal, dim=1)
    c = torch.sum(start_normal * normal, dim=1)
    s = torch.norm(v, dim=1)

    vx = torch.zeros((start_normal.shape[0], 3, 3), device=start_normal.device, dtype=start_normal.dtype)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]

    # Compute the rotation matrix using Rodrigues' rotation formula
    I = torch.eye(3, device=start_normal.device, dtype=start_normal.dtype).unsqueeze(0).repeat(start_normal.shape[0], 1, 1)
    R = I + vx * s.unsqueeze(1).unsqueeze(2) + torch.bmm(vx, vx) * ((1 - c).unsqueeze(1).unsqueeze(2))
    return R


def build_quaternion_from_rotation(rot):
    w = torch.sqrt(1 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]) / 2
    x = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * w)
    y = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * w)
    z = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * w)
    return torch.stack((x, y, z, w), dim=1)

def build_rotation_from_quaternion(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None):
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    with torch.no_grad():
        shape = depth.shape
        depth = depth.detach().reshape(-1, 1, *shape[-2:])
        if mask is not None:
            mask = mask.reshape(-1, 1, *shape[-2:])

        if mask is None:
            diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
        else:
            diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

        edge = torch.zeros_like(depth, dtype=torch.bool)
        if atol is not None:
            edge |= diff > atol
        if rtol is not None:
            edge |= (diff / depth).nan_to_num_() > rtol
        edge = edge.reshape(*shape)
        return edge

@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    scales_mul: Float[Tensor, "*batch 2"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    normals: Float[Tensor, "*batch 3"]

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    scl_mul_range: float = 3.0


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((2, 4, 3 * self.d_sh), dim=-1)
        scales_mul = torch.ones_like(scales)
        scales_z = self.cfg.gaussian_scale_min * torch.ones_like(scales[..., :1])
        scales = torch.cat((scales, scales_z), dim=-1)
        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        opacities_max = 0.6
        opacities = opacities * opacities_max

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
            scales_mul=scales_mul,
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 6 + 3 * self.d_sh



class SurfelAdapter(GaussianAdapter):
    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        
        scales_mul, sh = raw_gaussians.split((2, 3 * self.d_sh), dim=-1)
        scales_mul_range = self.cfg.scl_mul_range
        scales_mul = torch.clamp(F.softplus(scales_mul),
            min=1/scales_mul_range,
            max=scales_mul_range,
        )

        assert input_images is not None
        b,v,c,h,w = input_images.shape
        # Normalize the quaternion features to yield a valid quaternion.
        # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        means = rearrange(means, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w)
        rotations, scales, normals = pts2covariance(means)
        
        scales_mul = rearrange(scales_mul, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w)
        scales = scales * scales_mul

        scales_z = self.cfg.gaussian_scale_min * torch.ones_like(scales[..., :1])
        scales = torch.cat((scales, scales_z), dim=-1)
        
        scales = torch.clamp(scales,
        min=self.cfg.gaussian_scale_min,
        max=self.cfg.gaussian_scale_max,
        )

        covariances = (rotations
            @ scales.diag_embed()
            @ rearrange(scales.diag_embed(), "... i j -> ... j i")
            @ rearrange(rotations, "... i j -> ... j i"))
        quats = build_quaternion_from_rotation(rotations.view(b*v*h*w, c, c))
        quats = quats.view(b, v, h, w, 4)
        
        means = rearrange(means, "b v h w c -> b v (h w) () () c")
        quats = rearrange(quats, "b v h w c -> b v (h w) () () c")
        scales = rearrange(scales, "b v h w c -> b v (h w) () () c")
        covariances = rearrange(covariances, "b v h w c1 c2 -> b v (h w) () () c1 c2")
        scales_mul = rearrange(scales_mul, "b v h w c -> b v (h w) () () c")
        normals = rearrange(normals, "b v h w c -> b v (h w) () () c")
                
        # Create world-space covariance matrices.
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        opacities_max = 0.6
        opacities = opacities * opacities_max
        
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            scales=scales,
            rotations=quats,
            scales_mul=scales_mul,
            normals=normals,
        )
        
    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2 

    @property
    def d_in(self) -> int:
        return 2 + 3 * self.d_sh