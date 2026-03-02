<p align="center">
  <h1 align="center">SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/he-bing-2aab86346/">Bing He</a>
    ·
    <a href="https://g-1nonly.github.io">Jingnan Gao</a>
    ·
    <a href="https://github.com/UnoC-727">Yunuo Chen</a>
    ·
    Ning Cao
    ·
    Gang Chen
    ·
    Zhengxue Cheng
    ·
    Li Song
    ·
    Wenjun Zhang
  </p>
  <h3 align="center">ICLR 2026</h3>
  <div align="center">
    <a href="https://arxiv.org/abs/2602.02000"><img src="https://img.shields.io/badge/arXiv-2602.02000-b31b1b.svg" alt="arXiv"></a>
    <a href="https://hebing-sjtu.github.io/SurfSplat-website/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
  </div>
</p>

---

### 📅 Roadmap
- [✅] Release arXiv preprint.
- [✅] Launch project website.
- [✅] Release training code.
- [ ] Release inference code and checkpoints.
- [ ] Release preprocessed scannet dataset.
- [ ] Release mesh extraction code.

---

## 🖼️ Methodology
*SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures.*

---

## Repository

First clone the repository.

```bash
# Clone this repo
git clone https://github.com/hebing-sjtu/SurfSplat.git --recursive
# or
git clone https://github.com/hebing-sjtu/SurfSplat.git
git submodule update --init --recursive
```

## Installation

Our code is developed using PyTorch 2.4.0, CUDA 12.4, and Python 3.10. 

We recommend setting up a virtual environment using either [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html) before installation:

```bash
# conda
conda create -y -n surfsplat python=3.10
conda activate surfsplat

# or venv
# python -m venv /path/to/venv/surfsplat
# source /path/to/venv/surfsplat/bin/activate

# installation
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# install rasterization
cd submodules/rasterization
pip install .
cd ...
```

---

## Model Zoo

Our pre-trained models will be hosted on Hugging Face 🤗.

---

## Datasets
### Train & Test

For RealEstate10K and ACID datasets, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat). [dcharatan](https://github.com/dcharatan) kindly provide the [preprocessed RealEstate10K and ACID datasets](http://schadenfreude.csail.mit.edu:8000/)

### Test only
For DTU dataset, we follow [MVSplat](https://github.com/donydchen/mvsplat). [donydchen](https://github.com/donydchen) kindly provide the [preprocessed DTU dataset](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)

For DL3DV dataset, we follow [DepthSplat](https://github.com/cvg/depthsplat). We will provided preprocessed dataset for test.

For Scannet dataset, We will provided preprocessed dataset for test.

---

## Training

- Before training, you need to download the pre-trained [UniMatch](https://github.com/autonomousvision/unimatch) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) weights, and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.

```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
```

- Check out the scripts [scripts/re10k_train.sh](scripts/re10k_train.sh) for details.

---

## Evaluation

- Check out the scripts [scripts/dataset_test.sh](scripts/dataset_test.sh) for details.


---


## Camera Conventions

The camera intrinsic matrices are normalized, with the first row divided by the image width and the second row divided by the image height.

The camera extrinsic matrices follow the OpenCV convention for camera-to-world transformation (+X right, +Y down, +Z pointing into the screen).


---

## ✒️ Citation
If you find our work helpful, please consider citing:

```bibtex
@misc{he2026surfsplatconqueringfeedforward2d,
      title={SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors}, 
      author={Bing He and Jingnan Gao and Yunuo Chen and Ning Cao and Gang Chen and Zhengxue Cheng and Li Song and Wenjun Zhang},
      year={2026},
      eprint={2602.02000},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.02000}, 
}