# LiCROcc

[![arXiv](https://img.shields.io/badge/ArXiv-2407.16197-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2407.16197) [![web](https://img.shields.io/badge/Web-LiCROcc-blue.svg?style=plastic)](https://hr-zju.github.io/LiCROcc/)  [![star](https://img.shields.io/github/stars/HR-zju/LiCROcc)](https://github.com/HR-zju/LiCROcc)

This repository contains the implementation of the paper.

If you find our work useful, Please give us a star 🌟!

> LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction using LiDAR and Camera <br>
> [Yukai Ma](https://april.zju.edu.cn/team/yukai-ma/)<sup>1,2</sup>, [Jianbiao Mei](https://april.zju.edu.cn/team/jianbiao-mei/)<sup>1,2</sup>, [Xuemeng Yang](https://scholar.google.com/citations?user=xGuZsikAAAAJ&hl=zh-CN)<sup>2</sup>, [Licheng Wen](https://wenlc.cn/)<sup>2</sup>, [Weihua Xu](https://person.zju.edu.cn/whxu)<sup>1</sup>, [Jiangning Zhang](https://zhangzjn.github.io/)<sup>1</sup>, [Botian Shi](https://scholar.google.com/citations?user=K0PpvLkAAAAJ)<sup>2,^</sup>, [Yong Liu](https://scholar.google.com.hk/citations?user=qYcgBbEAAAAJ&hl=zh-CN&oi=sra)<sup>1,^</sup>, [Xingxing Zuo](https://xingxingzuo.github.io/)<sup>3</sup><br>
> <sup>1</sup>ZJU <sup>2</sup>PJLab<sup>3</sup>TUM<br>
> <sup>^</sup>Corresponding Authors

## Plan

We will release the code for radar SSC in this repo.

## Getting Started
### Installation
Please refer to [SSC-RS](https://github.com/NVlabs/VoxFormer/tree/main)
- spconv-cu111==2.1.25  
- torch-scatter==2.0.8  
- torchmetrics>=0.9.0  


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
│   │   ├── nuscenes_occ_infos_train.pkl/
│   │   ├── nuscenes_occ_infos_val.pkl/
│   │   ├── radar_bev_filter/
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


Eval RC-LiCROcc
```bash
./tools/dist_test.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2LR123.py ./path/to/ckpts.pth N_GPUs
```

Eval R-LiCROcc
```bash
./tools/dist_test.sh ./projects/configs/ssc_rs/ssc_rs_base_nuscenes_LC2radar12.py ./path/to/ckpts.pth N_GPUs
```

## Model Zoo
coming soon

## Cite Us

```bibtex
@misc{ma2024licroccteachradaraccurate,
title={LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction using LiDAR and Camera}, 
author={Yukai Ma and Jianbiao Mei and Xuemeng Yang and Licheng Wen and Weihua Xu and Jiangning Zhang and Botian Shi and Yong Liu and Xingxing Zuo},
year={2024},
eprint={2407.16197},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2407.16197}, 
}  
```

## Credit

We adopt the following open-sourced projects:

- [SSC-RS](https://github.com/Jieqianyu/SSC-RS)
- [FlashOCC](https://github.com/Yzichen/FlashOCC)
- [MonoOcc](https://github.com/ucaszyp/MonoOcc)
