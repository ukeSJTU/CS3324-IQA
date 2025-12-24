# PyramidIQA

基于多尺度特征金字塔网络和注意力机制的无参考图像质量评估模型

[![Compile LaTeX Report](https://github.com/ukeSJTU/CS3324-IQA/actions/workflows/compile-latex.yml/badge.svg)](https://github.com/ukeSJTU/CS3324-IQA/actions/workflows/compile-latex.yml)

## 📄 项目报告

📖 [在线查看完整报告 (PDF)](https://ukesjtu.github.io/CS3324-IQA/final_report.pdf)

## 📝 项目简介

PyramidIQA 是一个改进的无参考图像质量评估（No-Reference IQA）模型，基于 HyperIQA 架构，通过引入多尺度特征提取、特征金字塔网络（FPN）和卷积注意力模块（CBAM）来提升图像质量评估的准确性和泛化能力。

### 主要特性

- **多尺度特征提取**：从 ResNet-50 的多个层级（layer2, layer3, layer4）提取特征，捕获不同空间尺度的质量信息
- **特征金字塔网络**：通过自顶向下的路径和横向连接，将高层语义信息融合到低层特征中
- **注意力增强**：集成 CBAM 模块，自适应地关注质量相关的通道和空间区域
- **组合损失函数**：结合 L1 损失和排序损失，优化绝对质量预测和相对排序

### 实验结果

在 KonIQ-10k 数据集上训练，并在多个基准数据集上进行评估：

| 数据集 | SRCC | PLCC |
|--------|------|------|
| KonIQ-10k (测试集) | 0.8988 | 0.9166 |
| SPAQ (跨数据集) | 0.8979 | 0.9054 |
| KADID-10K (跨数据集) | 0.8343 | 0.8669 |
| AGIQA-3K (跨数据集) | 0.6652 | 0.6968 |

相比基线 HyperIQA，在 SPAQ 跨数据集评估上提升了 1.58% 的 SRCC，验证了多尺度金字塔架构的有效性。

## 🚀 快速开始

### 环境要求

```bash
# 创建 conda 环境
conda create -n mxlyu_iqa python=3.8
conda activate mxlyu_iqa

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
cd ms_hyperIQA
bash scripts/train.sh
```

### 评估模型

```bash
cd ms_hyperIQA
bash scripts/evaluate.sh
```

### Demo 演示

```bash
cd ms_hyperIQA
bash scripts/run_demo.sh
```

## 📂 项目结构

```
├── hyperIQA/              # 基线 HyperIQA 实现
├── ms_hyperIQA/           # PyramidIQA (多尺度版本)
│   ├── models.py          # 模型架构定义
│   ├── dataset.py         # 数据集加载
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   ├── demo.py            # 演示脚本
│   └── scripts/           # 运行脚本
├── report/                # LaTeX 项目报告
│   └── final_report.tex   # 报告源文件
├── results/               # 实验结果和图表
└── diagrams/              # 架构图
```

## 📊 计算复杂度

- **额外计算成本**：+15.4% FLOPs（主要来自 FPN 操作）
- **参数增加**：+0.9%（额外 118K 参数）
- **推理效率**：在单张 GPU 上约 45ms/图像

## 🎓 课程信息

本项目是上海交通大学 CS3324 数字图像处理课程的大作业。

- **课程**：CS3324 数字图像处理
- **作者**：Mingxi Lyu (吕铭熙)
- **学号**：523030910081
- **时间**：2024年12月

## 📚 参考文献

- **HyperIQA**: Su et al., "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network", CVPR 2020
- **Feature Pyramid Networks**: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

## 📄 作业要求

原始课程作业要求请参见 [ASSIGNMENT.md](ASSIGNMENT.md)

## 📜 License

本项目的 HyperIQA 基线实现基于原始 [HyperIQA repository](https://github.com/SSL92/hyperIQA)。
