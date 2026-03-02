# Model Zoo

- We provide pre-trained models for view synthesis with 3D Gaussian splatting and scale-consistent depth estimation from multi-view posed images.

- We assume that the downloaded weights are stored in the `pretrained` directory. It's recommended to create a symbolic link from `YOUR_MODEL_PATH` to `pretrained` using
```
ln -s YOUR_MODEL_PATH pretrained
```

- To verify the integrity of downloaded files, each model on this page includes its [sha256sum](https://sha256sum.com/) prefix in the file name, which can be checked using the command `sha256sum filename`.


## Gaussian Splatting

- The models are trained on RealEstate10K (re10k) and/or DL3DV (dl3dv) datasets at resolutions of 256x256, 256x448, and 448x768. The number of training views ranges from 2 to 10.

- The "&rarr;" symbol indicates that the models are trained in two stages. For example, "re10k &rarr; (re10k+dl3dv)" means the model is firstly trained on the RealEstate10K dataset and then fine-tuned using a combination of the RealEstate10K and DL3DV datasets.


| Model                                                        |       Training Data        |  Training Resolution  | Training Views | Params (M) |                           Download                           |
| ------------------------------------------------------------ | :------------------------: | :-------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth         |           re10k            |        256x256        |       2        |     37     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth) |
| depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth          |           re10k            |        256x256        |       2        |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth) |
| depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth         |           re10k            |        256x256        |       2        |    360     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth) |
| depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth          |           re10k            |        256x448        |       2        |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth) |
| depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth    |     re10k &rarr; dl3dv     |        256x448        |      2-6       |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth) |
| depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth | re10k &rarr; (re10k+dl3dv) | 256x448 &rarr;448x768 |      4-10      |     37     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth) |
| depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth | re10k &rarr; (re10k+dl3dv) | 256x448 &rarr;448x768 |      2-6       |    117     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth) |



## Depth Prediction

- The depth models are trained with the following procedure:
  - Initialize the monocular feature with Depth Anything V2 and the multi-view Transformer with UniMatch.
  - Train the full DepthSplat model end-to-end on the mixed RealEstate10K and DL3DV datasets.
  - Fine-tune the pre-trained depth model on the depth datasets with ground truth depth supervision. The depth datasets used for fine-tuning include ScanNet, TartanAir, and VKITTI2.
- The depth models are fine-tuned with random numbers (2-8) of input images, and the training image resolution is 352x640.
- The scale of the predicted depth is aligned with the scale of camera pose's translation.

| Model                                                   |                  Training Data                   |  Training Resolution   | Training Views | Params (M) |                           Download                           |
| ------------------------------------------------------- | :----------------------------------------------: | :--------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| depthsplat-depth-small-352x640-randview2-8-e807bd82.pth | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2-8       |     36     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-small-352x640-randview2-8-e807bd82.pth) |
| depthsplat-depth-base-352x640-randview2-8-65a892c5.pth  | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2-8       |    111     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth) |


