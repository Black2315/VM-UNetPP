# VM-UNet

这是“VM-UNet: Vision Mamba UNet用于医学图像分割”项目的官方代码库 {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## 摘要

在医学图像分割领域，基于CNN和Transformer的模型已经得到了广泛的研究。然而，CNN在长程建模能力方面存在局限性，而Transformer则受到其二次计算复杂度的限制。最近，状态空间模型（SSM），以Mamba为代表，作为一种有前景的方法出现。它们不仅在建模长程交互方面表现出色，而且保持了线性计算复杂度。本文基于状态空间模型，提出了一种用于医学图像分割的U型架构模型——Vision Mamba UNet（VM-UNet）。具体来说，我们引入了视觉状态空间（VSS）模块作为基础模块，用于捕捉广泛的上下文信息，并构建了一个不对称的编码解码结构。我们在ISIC17、ISIC18和Synapse数据集上进行了全面的实验，结果表明，VM-UNet在医学图像分割任务中具有竞争力。据我们所知，这是第一个基于纯SSM模型构建的医学图像分割模型。我们的目标是建立一个基准，并为未来更高效、更有效的基于SSM的分割系统的发展提供有价值的见解。

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