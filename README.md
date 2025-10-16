## Overview

这是一些基础神经网络模型的复现，包括CNN（Lenet、ResNet）、RNN（LSTM、GRU）、Transformers等，供初学者参考…



## Installation and Usage

### Prerequisites

- 安装 [Anaconda](https://www.anaconda.com/)
- 安装 [NVIDIA GeForce Driver](https://www.nvidia.com/drivers/) (GPU 支持)
- 安装 [PyTorch（cudatoolkit、cuDNN）](https://pytorch.org/)

### Steps

1. 创建并激活特定 Python 解释器版本的 conda 环境

   ```bash
   conda create -n pytorch python==3.12
   conda env list
   conda activate pytorch
   conda list
   ```

2. 检查 NVIDIA 驱动安装，对齐 Pytorch 版本

   ```bash
   nvidia-smi
   ```

3. 安装对应版本 PyTorch 和相关包
   ```bash
   pip install -r requirements.txt
   ```

### Run

1. 激活 conda 环境：
   ```bash
   conda activate pytorch
   ```

2. 运行测试脚本验证环境配置：

   ```bash
   python scripts/torch_test.py
   python scripts/transformers_test.py
   ```

3. 导航到所需的模型目录并运行训练脚本：

   ```bash
   cd AlexNet  # 或其他任何模型目录
   python train.py
   ```

4. 完成后停用环境：
   ```bash
   conda deactivate
   ```



## Guide

### Works

#### Classification

##### LeNet-5

##### AlexNet

##### VGG-16

##### GoogLeNet

##### ResNet-18

##### DenseNet

#### NLP

##### LSTM

##### GRU

##### Transformer



## More

本仓库仅源代码，您可以通过阅读 [NN Basic | 龙犊&小窝🪹~](https://biglonglong.github.io/home/posts/know/nn-basic/) 了解各个模型的独特设计和优缺点。



## References

- [Pytorch框架与经典卷积神经网络与实战_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1e34y1M7wR/?spm_id_from=333.337.search-card.all.click)
- [手把手教学|快速带你入门深度学习与实战_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1eP411w7Re/?spm_id_from=333.1387.homepage.video_card.click&vd_source=1a278fe24f00dd5c69f2875b5add5a19)

