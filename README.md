# CS3324 数字图像处理大作业

## 任务概述

设计一个图像视觉感知质量评价(Perceptual-IQA)模型，不限于使用机器学习或深度学习方法，验证其在主流数据集上的性能。

### 参考方法（可选择复现）

-   **Stair-IQA**: https://github.com/sunwei925/StairIQA (基于深度神经网络)
-   **DBCNN**: https://github.com/zwx8981/DBCNN (基于深度神经网络)
-   **Hyper-IQA**: https://github.com/SSL92/hyperIQA (基于深度神经网络)

**注意**: 我们鼓励创新性的工作以及体现个人探索收获的项目报告撰写。不完全依赖参考方法、体现个人思考的工作可以得到额外分数。

---

## 主要任务（必须完成）

### 1. 数据集要求

-   **主要训练集**: KonIQ 数据集（已预划分训练/测试部分）
    -   训练部分可自行划分训练/验证集比例
-   **测试数据集**:
    -   KonIQ 测试集（主要评估）
    -   SPAQ（跨数据集测试）
    -   KADID-10K（跨数据集测试）
    -   AGIQA-3K（跨数据集测试）

### 2. 性能指标要求

**必须达到的性能指标**:

-   **KonIQ 测试集**: PLCC 和 SRCC 均需超过 **0.75**
-   **SPAQ 数据集**: PLCC 和 SRCC 均需超过 **0.70**
-   **KADID-10K 和 AGIQA-3K**: 性能指标不做强制要求（作为额外加分参考）

**评价指标说明**:

-   **SRCC** (Spearman Rank Correlation Coefficient，斯皮尔曼等级相关系数):

    $$\text{SRCC} = 1 - \frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}$$

    其中 $d_i$ 是第 $i$ 个样本的预测排名与真实排名之差，$n$ 是样本总数。

-   **PLCC** (Pearson Linear Correlation Coefficient，皮尔逊线性相关系数):

    $$\text{PLCC} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

    其中 $x_i$ 是预测质量分数，$y_i$ 是真实质量分数，$\bar{x}$ 和 $\bar{y}$ 分别是它们的均值。

### 3. 实现要求

-   可参考集成的 IQA 算法库（如 https://github.com/chaofengc/IQA-PyTorch）
-   最终提交的代码中需要完整体现使用过程

---

## 额外任务（可选完成）

### 1. 基于视觉-语言模型的方法

可参考以下工作：

-   **CLIP-IQA(CLIP-IQA+)**: https://github.com/IceClear/CLIP-IQA (基于视觉语言模型的零样本方法)
-   **LIQE**: https://github.com/zwx8981/LIQE (基于视觉语言模型的对比学习微调方法)
-   **Q-ALIGN**: https://github.com/Q-Future/Q-Align (基于多模态大模型微调的方法)
-   **QualiCLIP**: https://github.com/miccunifi/QualiCLIP (基于对比学习的微调方法)

### 2. 探究不同训练损失函数

可探索但不限于以下损失函数：

-   **MSE Loss** (均方误差损失):

    $$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{n=1}^{N} ||y_n - \hat{y}_n||_2^2$$

    其中 $y_n$ 是真实质量分数，$\hat{y}_n$ 是预测质量分数。

-   **SRCC Loss** (斯皮尔曼相关系数损失):

    $$\mathcal{L}_{\text{SRCC}} = 1 - \frac{\sum_n (v_n - \bar{v})(p_n - \bar{p})}{\sqrt{\sum_n (v_n - \bar{v})^2} \sqrt{\sum_n (p_n - \bar{p})^2}}$$

    其中 $v_n$ 和 $p_n$ 分别是预测和真实的排名。

-   **MAE Loss** (平均绝对误差损失):

    $$\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N} |Q_i - \hat{Q}_i|$$

-   **Rank Loss** (排序损失):

    $$\mathcal{L}_{\text{rank}}^{ij} = \max(0, |\hat{Q}_i - \hat{Q}_j| - e(\hat{Q}_i, \hat{Q}_j) \cdot (Q_i - Q_j))$$

    其中 $e(\hat{Q}_i, \hat{Q}_j) = \begin{cases} 1, & \hat{Q}_i \geq \hat{Q}_j \\ -1, & \hat{Q}_i < \hat{Q}_j \end{cases}$

-   **Pairwise-fidelity Loss** (需要用到意见分布):

    $$p^{\text{pred}}(A > B) = \Phi\left(\frac{\mu_i^{\text{pred}} - \mu_j^{\text{pred}}}{\sqrt{(\sigma_i^{\text{pred}})^2 + (\sigma_j^{\text{pred}})^2}}\right)$$

    其中 $\Phi$ 是标准正态分布的累积分布函数。

    -   参考论文: [Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](https://arxiv.org/pdf/2501.11561)
    -   参考论文: [No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency](https://arxiv.org/abs/2108.06858)

---

## 提交要求

### 1. 基本要求

-   **独立完成**: 课程报告必须独立完成，不允许小组合作
-   **学术诚信**: 严禁抄袭以及数据造假（会进行额外审查），违规将导致成绩取消

### 2. 报告格式

-   采用 **IEEE 短会议模板** (https://www.ieee.org/conferences/publishing/templates.html)
-   课程大作业材料中提供 LaTeX 模板，也可使用 Word
-   **除去引用文献部分，报告实际内容不少于三页**

### 3. 报告必须包含内容

无论使用自己设计的或给定参考模型的方法，报告均需包含：

1. **模型结构**

    - 模型的结构图或流程图
    - 对应的文本或伪代码介绍

2. **代码说明**

    - 模型代码中主要函数的作用介绍（可在代码中以注释形式呈现）

3. **实验结果**

    - 至少一个包含主要实验结果的表格
    - 至少包含主要任务要求报告的指标

4. **实验设定与训练日志**

    - 详细的实验超参数
    - 每轮的损失函数值
    - 验证集（如果有）的性能
    - 训练过程中随着步数增加的训练 LOSS 曲线

5. **计算复杂度分析**

    - 给定一张标准图片（会提供在大作业材料中）
    - 模型得出预测分数的运算量（以 FLOPS 记）
    - 吞吐时间

6. **实验结论和分析**

    - 最终的实验结论
    - 详细的结果分析

7. **可选任务说明**（如有完成）
    - 相关的实验细节
    - 对应的分析

### 4. 额外说明

-   如果直接选用参考模型完成，报告中需要提供更多的分析
-   个人设计的方法，建议与现有算法进行对比实验分析

### 5. 提交方式

-   **提交内容**: 课程报告（PDF 版）+ 完整的项目代码
-   **提交形式**: 压缩包
-   **命名格式**: `DIP课程项目+姓名+学号`
-   **提交渠道**:
    -   CANVAS 平台
-   **截止时间**: **2024 年 12 月 25 日 23:59**（15 周周四）
-   **不允许晚于截止期限提交**
