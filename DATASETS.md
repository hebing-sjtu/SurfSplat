# Datasets

For view synthesis experiments with Gaussian splatting, we mainly use [RealEstate10K](https://google.github.io/realestate10k/index.html) and [DL3DV](https://github.com/DL3DV-10K/Dataset) datasets. We provide the data processing scripts to convert the original datasets to pytorch chunk files which can be directly loaded with this codebase. 

Expected folder structure:

```
├── datasets
│   ├── re10k
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── dl3dv
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
```

It's recommended to create a symbolic link from `YOUR_DATASET_PATH` to `datasets` using
```
ln -s YOUR_DATASET_PATH datasets
```

Or you can specify your dataset path with `dataset.roots=[YOUR_DATASET_PATH]/re10k` and `dataset.roots=[YOUR_DATASET_PATH]/dl3dv` in the config.

We also provide instructions to convert additional datasets to the desired format.


## RealEstate10K

For experiments on RealEstate10K, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to train and evaluate on the 256x256 resolution.

Please refer to [pixelSplat repo](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p (360x640) dataset.

If you would like to train and evaluate on the high-resolution RealEstate10K dataset, you will need to download the 720p (720x1280) version. Please refer to [here](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download) for the downloading script. Note that the script by default downloads the 360p videos, you will need to modify the`360p` to `720p` in [this line of code](https://github.com/yilundu/cross_attention_renderer/blob/master/data_download/generate_realestate.py#L137) to download the 720p videos.

After downloading the 720p dataset, you can use the scripts [here](https://github.com/dcharatan/real_estate_10k_tools/tree/main/src) to convert the dataset to the desired format in this codebase.

Considering the full 720p dataset is quite large and may take time to download and process, we provide a preprocessed subset in `.torch` chunks ([download](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/re10k_720p_test_subset.zip)) containing two test scenes to quickly run inference with our model.

## DL3DV

For experiments on DL3DV, we primarily train and evaluate at a resolution of 256×448. Additionally, we train high-resolution models (448×768) for qualitative results.

For the test set, we use the [DL3DV-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark) split, which contains 140 scenes for evaluation. You can first use the script [src/scripts/convert_dl3dv_test.py](src/scripts/convert_dl3dv_test.py) to convert the test set, and then run [src/scripts/generate_dl3dv_index.py](src/scripts/generate_dl3dv_index.py) to generate the `index.json` file for the test set.

For the training set, we use the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset (270x480 resolution), where the 140 scenes in the test set are excluded during processing the training set. After downloading the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset, You can first use the script [src/scripts/convert_dl3dv_train.py](src/scripts/convert_dl3dv_train.py) to convert the training set, and then run [src/scripts/generate_dl3dv_index.py](src/scripts/generate_dl3dv_index.py) to generate the `index.json` file for the training set.

Please note that you will need to update the dataset paths in the aforementioned processing scripts.

If you would like to train and evaluate on the high-resolution DL3DV dataset, you will need to download the [DL3DV-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) version (540x960 resolution). Simply follow the same procedure for data processing, but update the `images_8` folder to `images_4`.

Please follow the [DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md) if you use this dataset in your project and kindly [reference the DL3DV paper](https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#bibtex).

Considering the full 960p dataset is quite large and may take time to download and process, we provide a preprocessed subset in `.torch` chunks ([download](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv_960p_test_subset.zip)) containing two test scenes to quickly run inference with our model. Please note that this released subset is intended solely for research purposes. We disclaim any responsibility for the misuse, inappropriate use, or unethical application of the dataset by individuals or entities who download or access it. We kindly ask users to adhere to the [DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md).


## ACID


We also evaluate our generalization on the [ACID](https://infinite-nature.github.io/) dataset. Note that we do not use the training set; you only need to [download the test set](http://schadenfreude.csail.mit.edu:8000/re10k_test_only.zip) (provided by [pixelSplat repo](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets)) for evaluation.



## Additional Datasets

If you would like to train and/or evaluate on additional datasets, just modify the [data processing scripts](src/scripts) to convert the dataset format. Kindly note the [camera conventions](https://github.com/cvg/depthsplat/tree/main?tab=readme-ov-file#camera-conventions) used in this codebase.

