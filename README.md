# Project Coding Basis

欢迎来到 Project Coding Basis 项目！这个项目是为了2024 ISBD in RUC编程基础课程建立的。

## 项目简介

Project Coding Basis 是一个基于深度学习的项目，包含了变分自编码器（VAE）和扩散模型的实现。该项目旨在理解和实现这些模型，并应用于图像生成任务。

## 环境管理

本项目使用 [uv](https://docs.astral.sh/uv/) 进行虚拟环境管理。

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
#### 调参
在训练模型时，您可以通过修改配置文件 `config.py` 来调整模型的超参数。以下是一些常见的超参数及其说明：

- `learning_rate`：学习率，控制模型参数更新的步长。
- `batch_size`：批次大小，决定每次迭代时使用的样本数量。
- `num_epochs`：训练轮数，决定模型训练的总迭代次数。
- `latent_dim`：潜在空间维度，决定VAE模型的潜在变量维度。

您可以在 `config.py` 文件中找到这些超参数的默认值，并根据需要进行调整。例如：

```python
# config.py
learning_rate = 0.001
batch_size = 64
num_epochs = 100
latent_dim = 20
```



修改这些参数后，重新运行训练脚本即可应用新的超参数设置：

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
