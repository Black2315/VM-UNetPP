# VM-UNet+++

这是“VM-UNet++: 多尺度密集跳跃连接的Vision Mamba UNet”项目的官方代码库 

## 摘要

自2023年12月提出以来，基于状态空间模型(SSM)的Mamba模型已经引起了广泛关注。基于Mamba的视觉模型有潜力取代传统的基于CNN和Transformer的模型，成为下一代分割模型。然而，现有的视觉Mamba模型在融合不同尺度的特征方面仍面临挑战。为了解决这一挑战，我们提出了VM-UNet++，这是一个基于视觉Mamba的重新设计的U型网络。该模型在编码器和解码器之间引入了一系列模块。这些基于Mamba架构的中间模块能够结合来自不同编码阶段的特征图的低层次和高层次细节。VM-UNet++在ISIC18、Synapse和ACDC数据集上进行了评估，在多个评估指标上都优于现有的主流架构。

## 0. 主要环境

```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

causal_conv1d和mamba_ssm的.whl文件可以在这里找到。{[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1.  准备数据集

### ISIC 数据集

- ISIC17和ISIC18数据集已按7:3的比例划分，可以在此处找到 {[百度网盘](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) 或 [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}。

  下载数据集后，需将其放入'./data/isic17/'和'./data/isic18/'文件夹，文件格式参考如下（以ISIC17数据集为例）。

  './data/isic17/'

  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png


### Synapse 数据集

- 对于Synapse数据集，可以参考[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)下载，或者通过[百度网盘](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)下载。

  下载数据集后，需将其放入'./data/Synapse/'，文件格式参考如下。

  './data/Synapse/'

  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz


## 2. 准备预训练权重

- 预训练的VMamba权重可以从[这里](https://github.com/MzeroMiko/VMamba)或[百度网盘](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy)下载。下载后，需将预训练权重存放在'./pretrained_weights/'目录下。



## 3. 训练VM-UNet

```bash
cd VM-UNet
python train.py  # Train and test VM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test VM-UNet on the Synapse dataset.
```

## 4. 获取输出结果

- 训练完成后，可以在'./results/'目录下获取结果

## 5. 致谢

- 感谢[VMamba](https://github.com/MzeroMiko/VMamba)和[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)的作者提供开源代码.