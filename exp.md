## SGDM (ISPRS-JPRS 2025)
Officical code for "[Semantic Guided Large Scale Factor Remote Sensing Image Super-resolution with Generative Diffusion Prior](https://www.sciencedirect.com/science/article/pii/S0924271624004714)", **ISPRS-JPRS**, 2025

<p align="center">
    <img src="assets/architecture.png" style="border-radius: 15px">
</p>

**PS: we have opened source the model parameters and configuration in the absence of vector maps. Welcome to quote and compare**

## :book:Table Of Contents

- [Visual Results](#visual_results)
- [Installation](#installation)
- [Pretrained Models](#pretrained_models)
- [Dataset](#dataset)
- [Inference](#inference)
- [Train](#train)

## <a name="visual_results"></a>:eyes:Visual Results

<!-- <details close>
<summary>General Image Restoration</summary> -->
### Results on synthetic dataset

<img src="assets/visual_results/sync_qualitative.png"/>

### Results on real-world dataset

<img src="assets/visual_results/real_qualitative.png"/>

### Results for style guidance

<img src="assets/visual_results/style-guidance.png"/>

### Results for style sampling

<img src="assets/visual_results/style-sample.png"/>

## <a name="installation"></a>:gear:Installation
```shell
# clone this repo
git clone https://github.com/wwangcece/SGDM.git

# create an environment with python >= 3.9
conda create -n SGDM python=3.9
conda activate SGDM
pip install -r requirements.txt
```

## <a name="pretrained_models"></a>:dna:Pretrained Models

[MEGA](https://mega.nz/folder/DUoFyDAb#Hf6u9z57-aiLr5RLL5j_ZA)

Download the model and place it in the checkpoints/ folder

## <a name="dataset"></a>:bar_chart:Dataset

[Google Drive](https://drive.google.com/file/d/1HIrHj1qurTTuRyUpYNZRxbmpjUfdO6dN/view?usp=drive_link)

For copyright reasons, we can only provide the geographic sampling points in the data and the download scripts of the remote sensing images. To download vector maps, you need to register a [maptiler](https://www.maptiler.com/) account and subscribe to the package.

In order to make it easier for you to understand our method, we have provided some toy examples for you to use! [Google Drive](https://drive.google.com/file/d/1AMKDjz9TdfJkORsw23T6hz_-rx4y6y_W/view?usp=drive_link).

## <a name="inference"></a>:crossed_swords:Inference

<a name="general_image_inference"></a>
First please modify the validation data set configuration files at configs/dataset

#### Inference for synthetic dataset

```shell
python inference_refsr_batch_simu.py \
--ckpt checkpoints/SGDM-syn.ckpt \
--config configs/model/refsr_simu.yaml \
--val_config configs/dataset/reference_sr_val_simu.yaml \
--output path/to/your/outpath \
--steps 50 \
--device cuda:0
```
Meanwhile, our model also supports inference in the absence of vector maps. All you need to do is set the use_map of the model configuration file (configs/model/refsr_simu.yaml) to False and inference using the sync_x32_no_map.ckpt pre-training weight.

#### Inference for real-world dataset

For style sampling
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--sample_style true \
--ckpt_flow_mean checkpoints/flow_mean \
--ckpt_flow_std checkpoints/flow_std \
--output path/to/your/outpath \
--steps 50 \
--device cuda:0
```

For style guidance
```shell
python inference_refsr_batch_real.py \
--ckpt checkpoints/SGDM-real.ckpt \
--config configs/model/refsr_real.yaml \
--val_config configs/dataset/reference_sr_val_real.yaml \
--output 50 path/to/your/outpath \
--steps 50 \
--device cuda:0
```

## <a name="train"></a>:stars:Train
Firstly load pretrained SD parameters:
```shell
python scripts/init_weight_refsr.py \
--cldm_config configs/model/refsr_simu.yaml \
--sd_weight checkpoints/v2-1_512-ema-pruned.ckpt \
--output checkpoints/init_weight/init_weight-refsr-simu.pt
```
Secondly please modify the training configuration files at configs/train_refsr_simu.yaml.
Finally you can start training:
```shell
python train.py \
--config configs/train_refsr_simu.yaml
```

For training SGDM+ (real-world version), you just need to replace the model configuration file with configs/model/refsr_real.yaml and the training configuration file with configs/train_refsr_real.yaml. And one more step is to train the style normalizing flow model.
Firstly please collect all the style vectors in the training dataset for model training:
```shell
python model/Flows/save_mu_sigama.py \
--ckpt path_to_saved_ckpt \
--data_config configs/dataset/reference_sr_train_real.yaml \
--savepath model/Flows/results \
--device cuda:1
```
Then you can train the  style normalizing flow model:
```shell
python model/Flows/mu_sigama_estimate_normflows_single.py
```
The saved model parameters can be used for style sampling.

## Citation

Please cite us if our work is useful for your research.

```
@article{wang2025semantic,
  title={Semantic guided large scale factor remote sensing image super-resolution with generative diffusion prior},
  author={Wang, Ce and Sun, Wanjie},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={220},
  pages={125--138},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement

This project is based on [Diffbir](https://github.com/XPixelGroup/DiffBIR). Thanks for their awesome work.

## Contact
If you have any questions, please feel free to contact with me at cewang@whu.edu.cn.
