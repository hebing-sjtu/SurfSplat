import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm


# https://github.com/autonomousvision/unimatch/blob/master/utils/visualization.py


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def viz_depth_tensor(disp, return_numpy=False, return_float=False, colormap="plasma"):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.cpu().numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(
        np.uint8
    )  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]
    
    if return_float:
        viz = viz.float() / 255.0
    return viz



def viz_normal_tensor(normal, return_numpy=False, return_float=False):
    """
    Visualizes surface normal maps. 
    Maps values from [-1, 1] to [0, 255] (RGB).
    
    Args:
        normal (torch.Tensor): Input tensor of shape (N, 3, H, W) or (3, H, W).
        return_numpy (bool): If True, returns numpy array [N, H, W, 3] or [H, W, 3] (uint8).
        return_float (bool): If True, returns tensor in range [0, 1] (float). 
                             Default returns tensor in range [0, 255] (uint8).
    """
    assert isinstance(normal, torch.Tensor)
    normal = F.normalize(normal, dim=1, p=2)
    
    vis_data = normal.detach().permute(0, 2, 3, 1).cpu().numpy()[0]
    vis_data = (vis_data + 1.0) * 0.5 * 255.0
    vis_data = np.clip(vis_data, 0, 255).astype(np.uint8) # [H, W, 3]
    
    if return_numpy:
        return vis_data

    # 5. Convert back to Tensor
    viz = torch.from_numpy(vis_data).permute(2, 0, 1) # [3, H, W]

    if return_float:
        viz = viz.float() / 255.0

    return viz