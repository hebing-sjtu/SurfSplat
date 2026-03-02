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
INPUT_IMAGE_DIR = Path("/home/bing/FF-GS/depthsplat_v2/datasets/tandt")
OUTPUT_DIR = Path("/home/bing/FF-GS/depthsplat_v2/datasets/tandt_chunks")

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


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    img_path = example_path / "images"
    return {
        int(path.stem): load_raw(path)
        for path in img_path.iterdir()
        if path.suffix.lower() in [".jpg"]
    }


# === 定义 Metadata 和 Example ===
class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]

class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

import struct
import collections

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images



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
    intrinsic_path = example_path / "sparse/0" / "cameras.bin"
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
    target_shape = (546, 979)
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
                except KeyError:
                    print(f"Skipping {key} slice {slice_idx} because of missing images.")
                    start = end - 20  # 还是推进窗口
                    slice_idx += 1
                    continue
                assert len(slice_images) == len(slice_image_names), f"Mismatch: {key} slice {slice_idx}"
                
                indices = [image_names.index(i) for i in slice_image_names]
                # 构造 example 切片
                example_slice = {
                    "key": f"{key}_slice{slice_idx:02d}",
                    "url": f"{example['url']}_slice{slice_idx:02d}",
                    "images": slice_images,
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

        if chunk_size > 0:
            save_chunk()

        # 生成索引
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)