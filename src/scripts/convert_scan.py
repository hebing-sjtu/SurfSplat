import json
import subprocess
from PIL import Image

from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
# import cv2
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import os

from glob import glob

# === 路径设置 ===
INPUT_IMAGE_DIR = Path("/media/ps/ssd6/bing/scannet/test")
OUTPUT_DIR = Path("/media/ps/ssd6/bing/scannet_chunks2")

TARGET_BYTES_PER_CHUNK = int(2e8)  # 200MB一个chunk

# === 获取所有场景 key ===
def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    keys = []
    subdir = INPUT_IMAGE_DIR

    # iterate through all the subdirectories
    for key in subdir.iterdir():
        if key.is_dir():
            item = key.name.split('/')[-1]
            # item = '/'.join(['scans', item])
            print(item)
            keys.append(item)

    keys.sort()
    return keys


# === 读取文件大小 ===
def get_size(path: Path) -> int:
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))

# === 读取图片为原始bytes ===
# def load_rgb_images(example_path: Path) -> dict[str, Tensor]:
    # """
    # 读取指定路径下的 RGB 图像，并以字节流形式返回。

    # 参数:
    #     example_path (Path): 包含图像的目录路径。

    # 返回:
    #     Dict[str, Tensor]: 键为图像文件名（不含扩展名），值为对应的字节流张量。
    # """
    # images = {}
    # color_path = example_path / "color"
    # for path in color_path.glob("*.jpg"):
    #     with Image.open(path) as img:
    #         # 缩放图像至 640×480
    #         img_resized = img.resize((640, 480))
    #         byte_data = np.asarray(img_resized, dtype=np.uint8).tobytes()
    #         byte_tensor = torch.from_numpy(np.copy(np.frombuffer(byte_data, dtype=np.uint8)))
    #         images[path.stem] = byte_tensor
    # return images

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    img_path = example_path / "color"
    return {
        int(path.stem): load_raw(path)
        for path in img_path.iterdir()
        if path.suffix.lower() in [".jpg"]
    }

def load_depth_images(example_path: Path) -> dict[int, torch.Tensor]:
    """
    读取指定路径下的 16 位深度图，并转换为浮点数表示的深度值（以米为单位）。

    参数:
        example_path (Path): 包含深度图的目录路径。

    返回:
        Dict[str, torch.Tensor]: 键为图像文件名（不含扩展名），值为对应的深度图张量。
    """
    depth_images = {}
    depth_path = example_path / "depth"
    for path in depth_path.glob("*.png"):
        byte_tensor = load_raw(path)
        depth_images[int(path.stem)] = byte_tensor
    return depth_images

# === 定义 Metadata 和 Example ===
class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]

class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

# === 加载 intrinsic ===
def load_intrinsic(file_path: Path) -> np.ndarray:
    intrinsic_matrix = np.loadtxt(file_path, dtype=np.float32)

    # 动态获取图像分辨率
    w=640
    h=480

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    intrinsic = [fx / w, fy / h, cx / w, cy / h, 0.0, 0.0]
    return np.array(intrinsic, dtype=np.float32)


# === 加载 metadata (timestamps + cameras) ===
def load_metadata(example_path: Path) -> Metadata:
    
    url = str(example_path).split("/")[-1]
    intrinsic_path = example_path / "intrinsic" / "intrinsic_depth.txt"
    intrinsic = load_intrinsic(intrinsic_path)

    timestamps = []
    cameras = []

    # 先列出所有 .txt (排除 intrinsic.txt)
    pose_folder = example_path / "pose"
    pose_files = sorted([p for p in pose_folder.glob("*.txt")])
    for pose_file in pose_files:
        frame_id = int(pose_file.stem)
        c2w = np.loadtxt(pose_file, dtype=np.float32) 
        if not np.isfinite(c2w).all():
            print(f"Warning: NaN or inf in {pose_file}")
            continue  # skip this pose
        # print("c2w =\n", c2w)
        # print("scan2opencv =\n", scan2opencv)   
        # opencv_c2w = c2w @ scan2opencv
        opencv_c2w = c2w
        w2c = np.linalg.inv(opencv_c2w)[:3].flatten().tolist() # 转换为 OpenCV 的 w2c
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)
        timestamps.append(frame_id)
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url, 
        "timestamps": timestamps,
        "cameras": cameras,
    }

def is_image_shape_matched(image_dir, target_shape):
    image_path = sorted(glob(str(image_dir / "*")))
    if len(image_path) == 0:
        return False

    image_path = image_path[0]
    try:
        im = Image.open(image_path)
    except:
        return False
    w, h = im.size
    if (h, w) == target_shape:
        return True
    else:
        return False

def legal_check_for_all_scenes(root_dir, target_shape):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/nerfstudio")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        img_dir = os.path.join(sub_folder, "depth") 
        if not is_image_shape_matched(Path(img_dir), target_shape):
            print(f"image shape does not match for {sub_folder}")
            continue
        # pose_file = os.path.join(sub_folder, "transforms.json")
        # if not os.path.isfile(pose_file):
        #     print(f"cannot find pose file for {sub_folder}")
        #     continue

        valid_folders.append(sub_folder)

    return valid_folders


# === 主流程 ===
if __name__ == "__main__":
    target_shape = (480, 640)
    print("checking all scenes...")
    valid_scenes = legal_check_for_all_scenes(INPUT_IMAGE_DIR, target_shape)
    print("valid scenes:", len(valid_scenes))
    
    for stage in ["test"]:  # 如果需要 val/test，可以再加
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size, chunk_index, chunk

            chunk_key = f"{chunk_index:0>6}"
            print(f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB).")
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            if key == "Chunks":
                continue
            example_path = INPUT_IMAGE_DIR / key
            num_bytes = get_size(example_path)

            if not example_path.exists():
                print(f"Skipping {key} because it is missing.")
                continue

            # 读取 images 和 metadata
            images = load_images(example_path)
            depths = load_depth_images(example_path)
            example = load_metadata(example_path)

            # 根据 timestamps 找对应image
            image_names = [int(timestamp.item()) for timestamp in example["timestamps"]]
            n = len(image_names)
            start = 0
            slice_idx = 0
            
            while start < n:
                end = min(start + 350, n)
                if n - end < 100 and end != n:
                    end = n
                slice_image_names = image_names[start:end]
                try:
                    slice_images = [images[image_name] for image_name in slice_image_names]
                    slice_depths = [depths[image_name] for image_name in slice_image_names]
                except KeyError:
                    print(f"Skipping {key} slice {slice_idx} because of missing images.")
                    start = end - 20  # 还是推进窗口
                    slice_idx += 1
                    continue
                assert len(slice_images) == len(slice_image_names), f"Mismatch: {key} slice {slice_idx}"
                assert len(slice_depths) == len(slice_image_names), f"Mismatch: {key} slice {slice_idx}"
                
                indices = [image_names.index(i) for i in slice_image_names]
                # 构造 example 切片
                example_slice = {
                    "key": f"{key}_slice{slice_idx:02d}",
                    "url": f"{example['url']}_slice{slice_idx:02d}",
                    "images": slice_images,
                    "depths": slice_depths,
                    "timestamps": torch.tensor(slice_image_names, dtype=torch.int64),
                    "cameras": example["cameras"][indices],
                }
                
                chunk.append(example_slice)
                slice_length = end - start
                chunk_size += num_bytes * slice_length / n  
                print(f"    Added {key}_slice{slice_idx:02d} to chunk ({chunk_size / 1e6:.2f} MB).")

                if chunk_size >= TARGET_BYTES_PER_CHUNK:
                    save_chunk()
                    
                if not end == n:
                    start = end - 20  # overlap 20
                    slice_idx += 1
                else:
                    start = end
                    
            # try:
            #     example["images"] = [
            #         images[image_name] for image_name in image_names
            #     ]
            #     example["depths"] = [
            #         depths[image_name] for image_name in image_names
            #     ]
            # except KeyError:
            #     print(f"Skipping {key} because of missing images.")
            #     continue
            # assert len(example["images"]) == len(example["timestamps"]), f"Mismatch: {key}"
            # assert len(example["depths"]) == len(example["timestamps"]), f"Mismatch: {key}"  

            # example["key"] = key

            # print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            # chunk.append(example)
            # chunk_size += num_bytes

            # if chunk_size >= TARGET_BYTES_PER_CHUNK:
            #     save_chunk()

        if chunk_size > 0:
            save_chunk()

        # # 生成索引
        # print("Generate key:torch index...")
        # index = {}
        # stage_path = OUTPUT_DIR / stage
        # for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
        #     if chunk_path.suffix == ".torch":
        #         chunk = torch.load(chunk_path)
        #         for example in chunk:
        #             index[example["key"]] = str(chunk_path.relative_to(stage_path))
        # with (stage_path / "index.json").open("w") as f:
        #     json.dump(index, f)