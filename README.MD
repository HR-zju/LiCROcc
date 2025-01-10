# LiCROcc

[![arXiv](https://img.shields.io/badge/ArXiv-2407.16197-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2407.16197) [![web](https://img.shields.io/badge/Web-LiCROcc-blue.svg?style=plastic)](https://hr-zju.github.io/LiCROcc/)  [![star](https://img.shields.io/github/stars/HR-zju/LiCROcc)](https://github.com/HR-zju/LiCROcc)

This repository contains the implementation of the paper.

If you find our work useful, Please give us a star 🌟!

> LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction using LiDAR and Camera <br>
> [Yukai Ma](https://april.zju.edu.cn/team/yukai-ma/)<sup>1,2</sup>, [Jianbiao Mei](https://april.zju.edu.cn/team/jianbiao-mei/)<sup>1,2</sup>, [Xuemeng Yang](https://scholar.google.com/citations?user=xGuZsikAAAAJ&hl=zh-CN)<sup>2</sup>, [Licheng Wen](https://wenlc.cn/)<sup>2</sup>, [Weihua Xu](https://person.zju.edu.cn/whxu)<sup>1</sup>, [Jiangning Zhang](https://zhangzjn.github.io/)<sup>1</sup>, [Xingxing Zuo](https://xingxingzuo.github.io/)<sup>3</sup>, [Botian Shi](https://scholar.google.com/citations?user=K0PpvLkAAAAJ)<sup>2,^</sup>, [Yong Liu](https://scholar.google.com.hk/citations?user=qYcgBbEAAAAJ&hl=zh-CN&oi=sra)<sup>1,^</sup><br>
> <sup>1</sup>ZJU <sup>2</sup>PJLab<sup>3</sup>TUM<br>
> <sup>^</sup>Corresponding Authors

## News

🎉 [2024/11/06] LiCROcc was accepted by RAL!!!
## Getting Started
### Installation
```bash
# Create a conda virtual environment and activate it.
conda create -n licrocc python=3.8 -y
conda activate licrocc
# Install torch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f ttps://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# Install mmdet3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
# 
cd ../projects
pip install -v -e .
``````

### Prepare Dataset
-Please refer to [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) to prepare nuScenes dataset.
- Please refer to [CRN](https://github.com/youngskkim/CRN) to generate radar point cloud in BEV view.
```bash
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
```
Folder structure:
```bash
LiCROcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
│   │   ├── radar_bev_filter/
│   ├── nuScenes-Occupancy/
│   ├── depth_gt/
```

### Run and Eval
Download Teacher model [here](https://drive.google.com/file/d/11oejqfgTr489EeW-nftxMi2d6Rp07yRS/view?usp=drive_link)

Train RC-LiCROcc
```bash
./tools/dist_train.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2LR123.py N_GPUs
```

Train R-LiCROcc
```bash
./tools/dist_train.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2radar12.py N_GPUs
```
If an error is reported during training：
```bash
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```
Simply remove ‘verify=True’


Eval RC-LiCROcc
```bash
./tools/dist_test.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2LR123.py ./path/to/ckpts.pth N_GPUs
```

Eval R-LiCROcc
```bash
./tools/dist_test.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2radar12.py ./path/to/ckpts.pth N_GPUs
```

Visualization

Temporarily only support saving occupancy predictions (refer to [MonoScene](https://github.com/astra-vision/MonoScene#visualization) for visualization tools)
```bash
./tools/dist_test.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH
```

## Model Zoo
 Subset | Checkpoint | Logs | Note |
| :---: | :---: | :---: | :---: |
| RC-LiCROcc | [link](https://drive.google.com/file/d/14B9lfnVtrgWwmortPv3yydnIyav7AiFk/view?usp=drive_link)| [link](https://drive.google.com/file/d/1PLQ_ETfT_LLcZZvLrvMF3tgGpUp51NMG/view?usp=drive_link) | train on 8 A100|
| R-LiCROcc | [link](https://drive.google.com/file/d/1ETI3StsHwqQoBn4OkmELB8ppYTt7hunw/view?usp=drive_link)| [link](https://drive.google.com/file/d/1vvkoTrNSmBg_Ktd-yn0UjuFQp9it6DiO/view?usp=drive_link) | train on 8 A100|

## Cite Us

```bibtex
@ARTICLE{10777549,
  author={Ma, Yukai and Mei, Jianbiao and Yang, Xuemeng and Wen, Licheng and Xu, Weihua and Zhang, Jiangning and Zuo, Xingxing and Shi, Botian and Liu, Yong},
  journal={IEEE Robotics and Automation Letters}, 
  title={LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction Using LiDAR and Camera}, 
  year={2025},
  volume={10},
  number={1},
  pages={852-859},
  keywords={Radar;Semantics;Radar imaging;Three-dimensional displays;Laser radar;Feature extraction;Cameras;Sensors;Meteorology;Point cloud compression;Sensor fusion;semantic scene completion;knowledge distillation},
  doi={10.1109/LRA.2024.3511427}}
```

## Credit

We adopt the following open-sourced projects:

- [SSC-RS](https://github.com/Jieqianyu/SSC-RS)
- [FlashOCC](https://github.com/Yzichen/FlashOCC)
- [MonoOcc](https://github.com/ucaszyp/MonoOcc)
