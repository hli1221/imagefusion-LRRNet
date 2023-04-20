# LRRNet: A Novel Representation Learning Guided Fusion Framework for Infrared and Visible Images

[Hui Li](https://hli1221.github.io/), [Tianyang Xu](https://xu-tianyang.github.io/), Xiao-Jun Wu*, Jiwen Lu, Josef Kittler  
Accetped by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), DOI: 10.1109/TPAMI.2023.3268209  
[paper](https://doi.org/10.1109/TPAMI.2023.3268209)  
[Arxiv](https://arxiv.org/abs/2304.05172)  
[Supplemental materials]


## Platform

Python 3.7  
Pytorch >= 1.8  

## Training Dataset

[KAIST](https://soonminhwang.github.io/rgbt-ped-detection/) (S. Hwang, J. Park, N. Kim, Y. Choi, I. So Kweon, Multispectral pedestrian detection: Benchmark dataset and baseline, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1037–1045.) is utilized to train LRRNet.

## VGG-16 modal
[google drive](https://drive.google.com/file/d/19vG7UPbumgElmul_r2-CBR2jp5dtI66a/view?usp=share_link)

## The difference between LRRNet and other architectures

<img src="https://github.com/hli1221/imagefusion-LRRNet/blob/main/framework/fig-new-architecture.png" width="600">

## Learnable LRR block

<img src="https://github.com/hli1221/imagefusion-LRRNet/blob/main/framework/llrr-blocks-new.png" width="600">

## LRRNet - Fusion framework

<img src="https://github.com/hli1221/imagefusion-LRRNet/blob/main/framework/lrrnet-fusion-framework.png" width="600">

## LLRR block for RGBT tracking - framework

<img src="https://github.com/hli1221/imagefusion-LRRNet/blob/main/framework/fig-tracking-lrrnet-new.png" width="600">


If you have any question about this code, feel free to reach me(lihui.cv@jiangnan.edu.cn, hui_li_jnu@163.com) 

# Citation

```
@article{li2021rfn,
  title={{LRRNet: A novel representation learning guided fusion framework for infrared and visible images}},
  author={Li, Hui and Xu, Tianyang and Wu, Xiao-Jun and Lu, Jiwen and Kittler, Josef},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note={DOI: 10.1109/TPAMI.2023.3268209},
  publisher={IEEE}
}
```



