# Project Coding Basis

欢迎来到 Project Coding Basis 项目！这个项目是为了2024 ISBD in RUC编程基础课程建立的。

## 项目简介

Project Coding Basis 是一个基于深度学习的项目，包含了变分自编码器（VAE）和扩散模型的实现。该项目旨在帮助学生理解和实现这些模型，并应用于图像生成任务。

## 文件结构

```
project_coding_basis/
├── __pycache__/
├── .gitignore
├── .python-version
├── checkpoints/
│   ├── diffusion_model.pth
│   ├── vae.pth
├── config.py
├── data/
│   ├── FashionMNIST/
│   │   ├── raw/
│   │       ├── t10k-images-idx3-ubyte
│   │       ├── t10k-labels-idx1-ubyte
│   │       ├── train-images-idx3-ubyte
│   ├── MNIST/
│       ├── raw/
│           ├── ...
├── data_loader.py
├── example.ipynb
├── LICENSE.txt
├── Models/
│   ├── __pycache__/
│   ├── diffusion.py
│   ├── vae.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── results/
│   ├── diffusion/
│   │   ├── diffusion_losses.pkl
│   ├── vae/
│       ├── vae_loss.csv
├── train_diffusion.py
├── train_vae.py
├── utils/
│   ├── __pycache__/
│   ├── utils.py
├── uv.lock
```

## 安装

请按照以下步骤安装和配置项目：

1. 克隆仓库：
    ```bash
    git clone https://github.com/CabbageSage/project_coding_basis.git
    ```
2. 进入项目目录：
    ```bash
    cd project_coding_basis
    ```
3. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 使用

### 训练模型

- 训练VAE模型：
    ```bash
    python train_vae.py
    ```

- 训练扩散模型：
    ```bash
    python train_diffusion.py
    ```

### 生成图像

在 `example.ipynb` 中提供了生成图像的示例代码。您可以运行该notebook来生成和可视化图像。

## 许可证

本项目采用 [MIT 许可证](LICENSE.txt)。

感谢您的关注和支持！