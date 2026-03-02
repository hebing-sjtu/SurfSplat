import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU to use
from pathlib import Path
from math import sin, cos, radians

import hydra
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate
import json
from tqdm import tqdm

from ..visualization.vis_depth import viz_depth_tensor
import os
from PIL import Image
from einops import repeat
# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image, save_cam_info
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic, render_cuda
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_c2w_numpy(c2w0: np.ndarray, c2w1: np.ndarray, N: int, device: torch.device):
    """
    c2w0, c2w1: (4,4) numpy arrays
    returns: list of N (4,4) numpy arrays (including endpoints), uniformly spaced t in [0,1]
    """
    c2w0 = np.array(c2w0.cpu())
    c2w1 = np.array(c2w1.cpu())
    
    assert c2w0.shape == (4,4) and c2w1.shape == (4,4)
    # 取平移向量
    t0 = c2w0[:3, 3]
    t1 = c2w1[:3, 3]

    # 取旋转矩阵（3x3）
    R0 = c2w0[:3, :3]
    R1 = c2w1[:3, :3]

    # 用 scipy Rotation / Slerp 做旋转插值
    rots = R.from_matrix([R0, R1])
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, rots)

    times = np.linspace(0.0, 1.0, N)
    interp_rots = slerp(times)                 # Rotation 对象数组
    interp_trans = np.outer(1 - times, t0) + np.outer(times, t1)  # (N,3) 线性插值

    poses = []
    for i in range(N):
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = interp_rots[i].as_matrix()
        M[:3, 3] = interp_trans[i]
        poses.append(M)
        
    poses = torch.from_numpy(np.array(poses)).to(device)  # (N,4,4)
    return poses

with open("assets/evaluation_index_re10k.json") as f:
    scene_cfgs = json.load(f)

angle_list = list(range(0,360,30))
linear_inter_N = 11
# supple scenes
SCENES = (
    # scene, context 1, context 2, far plane
    
    # ("33baf3e18e5d7256", 34, 151, 3.0, [80], 1.4, 19),
    # ("46502a6038bd288f", 0, 49, 3.0, [70], 1.4, 19),
    # ("12dce44829d88985", 70, 181, 3.0, [60], 1.4, 19),
    # ("32294ad73efca3db", 99, 210, 3.0, [80], 1.4, 19),
    # ("fc6f664a700121e9", 72, 164, 2.0, [95], 1.4, 19),


    ("33baf3e18e5d7256", 34, 151, 3.0, angle_list, (5, 10, 8, 80), 1.4, 19),
    ("fc6f664a700121e9", 72, 164, 2.0, angle_list, (5, 10, 5, 100), 1.4, 19),
    ("32294ad73efca3db", 99, 210, 3.0, angle_list, (5, 10, 8, 80), 1.4, 19),
    ("12dce44829d88985", 70, 181, 3.0, angle_list, (5, 10, 5, 60), 1.4, 19),
    ("46502a6038bd288f", 0, 49, 3.0, angle_list, (5, 10, 5, 70), 1.4, 19),
)


FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 8
LINE_WIDTH = 1.8
LINE_COLOR = [255, 0, 0]
POINT_DENSITY = 0.5
CIRCLE = False
INTERPOLATE = True

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    # Prepare the checkpoint for loading.
    state_dict = torch.load(cfg.checkpointing.load, map_location="cpu")

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper(
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    ).to(device)
    model_wrapper.load_state_dict(state_dict["state_dict"])
    model_wrapper.eval()

    for idx, (scene, *context_indices, far, angles, circle_info, line_width, cam_div) in enumerate(
        tqdm(SCENES)
    ):
        LINE_WIDTH = line_width
        # Create a dataset that always returns the desired scene.
        view_sampler_cfg = ViewSamplerArbitraryCfg(
            "arbitrary",
            2,
            2,
            context_views=list(context_indices),
            target_views=[0, 0],  # use [40, 80] for teaser
        )
        cfg.dataset.view_sampler = view_sampler_cfg
        cfg.dataset.overfit_to_scene = scene

        # Get the scene.
        dataset = get_dataset(cfg.dataset, "test", None)
        example = default_collate([next(iter(dataset))])
        example = apply_to_collection(example, Tensor, lambda x: x.to(device))

        # Generate the Gaussians.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"], False, visualization_dump=visualization_dump
        )
        
        gaussians = gaussians["gaussians"]
        
        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        # Transform means into camera space.
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        )
        means = homogenize_points(means)
        w2c = example["context"]["extrinsics"].inverse()[0]
        means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]

        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        mask = mask & (means[..., 2] < far * 2)

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
            )
            return element[mask][None]

        if CIRCLE:
            # disable opacities
            gaussians.opacities = torch.where(
                gaussians.opacities > 0.01,
                torch.full_like(gaussians.opacities, 0.999),
                torch.full_like(gaussians.opacities, 0.001),
            )  
            for angle in angles:
                a_len,a_pos,b_len,b_pos = circle_info
                pose = torch.eye(4, dtype=torch.float32, device=device)
                rotation = R.from_euler("xyz", [-15 + a_pos + a_len*sin(radians(angle)), - 90 + b_pos - b_len*cos(radians(angle)), 0], True).as_matrix()
                pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
                translation = torch.eye(4, dtype=torch.float32, device=device)
                # visual balance, 0.5x pyramid/frustum volume
                translation[2, 3] = far * (0.5 ** (1 / 3))
                pose = translation @ pose

                ones = torch.ones((1,), dtype=torch.float32, device=device)
                render_args = {
                    "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
                    "width": ones * far * 2,
                    "height": ones * far * 2,
                    "near": ones * 0,
                    "far": ones * far,
                    "image_shape": (1024, 1024),
                    "background_color": torch.zeros(
                        (1, 3), dtype=torch.float32, device=device
                    ),
                    "gaussian_means": trim(gaussians.means),
                    "gaussian_covariances": trim(gaussians.covariances),
                    "gaussian_sh_coefficients": trim(gaussians.harmonics),
                    "gaussian_opacities": trim(gaussians.opacities),
                    # "fov_degrees": 1.5,
                }

                # Render alpha (opacity).
                dump = {}
                alpha_args = {
                    **render_args,
                    "gaussian_sh_coefficients": torch.ones_like(
                        render_args["gaussian_sh_coefficients"][..., :1]
                    ),
                    "use_sh": False,
                }
                # alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

                # # Render (premultiplied) color.
                # color = render_cuda_orthographic(**render_args)[0]
                _, c_info = render_cuda_orthographic(**render_args)

                base = Path(f"point_clouds/re10k/{idx:0>6}_{scene}/angles")
                save_cam_info(c_info, base / f"angle_{angle:0>3}.pkl")
                continue
                # Render depths. Without modifying the renderer, we can only render
                # premultiplied depth, then hackily transform it into straight alpha depth,
                # which is needed for sorting.
                depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
                depth = depth.norm(dim=-1)
                depth_args = {
                    **render_args,
                    "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
                    "use_sh": False,
                }
                depth_premultiplied = render_cuda_orthographic(**depth_args)
                depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

                # Save the rendering for later depth-based alpha compositing.
                layers = [(color, alpha, depth)]

                # Figure out the intrinsics from the FOV.
                fx = 0.5 / (0.5 * dump["fov_x"]).tan()
                fy = 0.5 / (0.5 * dump["fov_y"]).tan()
                dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
                dump_intrinsics[0, 0] = fx
                dump_intrinsics[1, 1] = fy
                dump_intrinsics[:2, 2] = 0.5

                # Compute frustum corners for the context views.
                frustum_corners = unproject_frustum_corners(
                    example["context"]["extrinsics"][0],
                    example["context"]["intrinsics"][0],
                    torch.ones((2,), dtype=torch.float32, device=device) * far / cam_div,
                )
                camera_origins = example["context"]["extrinsics"][0, :, :3, 3]

                # Generate the 3D lines that have to be computed.
                lines = []
                for corners, origin in zip(frustum_corners, camera_origins):
                    for i in range(4):
                        lines.append((corners[i], corners[i - 1]))
                        lines.append((corners[i], origin))

                # Generate an alpha compositing layer for each line.
                for line_idx, (a, b) in enumerate(lines):
                    # Start with the point whose depth is further from the camera.
                    a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
                    b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
                    start = a if (a_depth > b_depth).all() else b
                    end = b if (a_depth > b_depth).all() else a

                    # Create the alpha mask (this one is clean).
                    start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
                    end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
                    alpha = draw_lines(
                        torch.zeros_like(color),
                        start_2d[None],
                        end_2d[None],
                        (1, 1, 1),
                        LINE_WIDTH,
                        x_range=(0, 1),
                        y_range=(0, 1),
                    )

                    # Create the color.
                    lc = torch.tensor(
                        LINE_COLOR,
                        dtype=torch.float32,
                        device=device,
                    )
                    color = draw_lines(
                        torch.zeros_like(color),
                        start_2d[None],
                        end_2d[None],
                        lc,
                        LINE_WIDTH,
                        x_range=(0, 1),
                        y_range=(0, 1),
                    )

                    # Create the depth. We just individually render points.
                    wh = torch.tensor((w, h), dtype=torch.float32, device=device)
                    delta = (wh * (start_2d - end_2d)).norm()
                    num_points = delta / POINT_DENSITY
                    t = torch.linspace(0, 1, int(num_points) + 1, device=device)
                    xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
                    depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
                    depth = repeat(depth, "p -> p c", c=3)
                    xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
                    depth = draw_points(
                        torch.ones_like(color) * 1e10,
                        xy,
                        depth,
                        LINE_WIDTH,  # makes it 2x as wide as line
                        x_range=(0, 1),
                        y_range=(0, 1),
                    )

                    layers.append((color, alpha, depth))

                # Do the alpha compositing.
                canvas = torch.ones_like(color)
                colors = torch.stack([x for x, _, _ in layers])
                alphas = torch.stack([x for _, x, _ in layers])
                depths = torch.stack([x for _, _, x in layers])
                index = depths.argsort(dim=0)
                colors = colors.gather(index=index, dim=0)
                alphas = alphas.gather(index=index, dim=0)
                t = (1 - alphas).cumprod(dim=0)
                t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
                image = (t * colors).sum(dim=0)
                total_alpha = (t * alphas).sum(dim=0)
                image = total_alpha * image + (1 - total_alpha) * canvas

                base = Path(f"videos/{cfg.wandb['name']}/{scene}/circle")
                save_image(image, f"{base}/angle_{angle:0>3}.png")

        if INTERPOLATE:
            far = 2.
            c2w0 = example["context"]["extrinsics"][0,0]
            c2w1 = example["context"]["extrinsics"][0,1]
            fw_poses = interpolate_c2w_numpy(c2w0, c2w1, linear_inter_N, device=device)
            bw_poses = torch.flip(fw_poses, dims=[0])
            bw_poses = bw_poses[1:-1]
            poses = torch.cat([fw_poses, bw_poses],dim=0)
            for i, pose in enumerate(poses):
                render_args = {
                    "extrinsics": repeat(pose,"...->() ..."),
                    "intrinsics": repeat(example["context"]["intrinsics"][0,0],"...->() ..."),
                    "near": repeat(example["context"]["near"][0,0],"->()"),
                    "far": repeat(example["context"]["far"][0,0],"->()"),
                    "image_shape": (1024, 1024),
                    "background_color": torch.zeros(
                        (1, 3), dtype=torch.float32, device=device
                    ),
                    "gaussian_means": gaussians.means,
                    "gaussian_covariances": gaussians.covariances,
                    "gaussian_sh_coefficients": gaussians.harmonics,
                    "gaussian_opacities": gaussians.opacities,
                    # "fov_degrees": 1.5,
                }

                # Render (premultiplied) color.
                # color = render_cuda(**render_args)[0]
                _, c_info = render_cuda(**render_args)

                base = Path(f"point_clouds/re10k/{idx:0>6}_{scene}/angles")
                save_cam_info(c_info, base / f"inter_{i:0>3}.pkl")
                continue
                base2 = Path(f"videos/{cfg.wandb['name']}/{scene}/inter")
                save_image(color, f"{base2}/inter_{i:0>3}.png")   
                
            a = 1
        if CIRCLE:
            continue
            os.system(
                f"ffmpeg -framerate 30 -pattern_type glob \
                    -i \"{base}/*.png\" -c:v libx264 -pix_fmt yuv420p \
                        {base}/aaa_{scene}_output.mp4 -y" \
            )
        if INTERPOLATE:
            continue
            os.system(
                f"ffmpeg -framerate 30 -pattern_type glob \
                    -i \"{base2}/*.png\" -c:v libx264 -pix_fmt yuv420p \
                        {base2}/aaa_{scene}_output.mp4 -y" \
            )
        a = 1
    a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
