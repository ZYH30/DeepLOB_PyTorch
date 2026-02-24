# DeepLOB-PyTorch: A Robust Quantitative Implementation

本项目是经典高频限价订单簿（LOB）预测模型 **DeepLOB** 的工业级 PyTorch 1:1 平行迁移与量化实战优化版本。在还原原论文通过卷积神经网络（CNN）提取空间微观结构、通过 Inception 与 LSTM 提取时序多尺度动量结构的基础上，针对高频时序数据的异方差性（Heteroskedasticity）与极端类别不平衡问题，引入了多项严谨的统计学与工程级优化。

## 📎 参考资源 (References)
* **原学术论文**: [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668)
* **公开数据源**: [Kaggle - Bitcoin Perpetual (BTCUSDT) LOB Data](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data)
* **TensorFlow 原始实现**: [Github - S-razmi/DeepLOB](https://github.com/S-razmi/DeepLOB)

---

## 🧠 核心处理逻辑 (Core Logic)

本架构将微观订单簿数据的预测转化为一个端到端的表征学习（Representation Learning）过程，核心数据流如下：

1. **输入表征 (State Representation)**: 
   截取历史 $T$ 个时间步的订单簿快照，每个时间步包含买卖双向各 10 档的价格与容量信息，构建张量 $X \in \mathbb{R}^{N \times 1 \times T \times 40}$。
2. **微观结构特征抽取 (Spatial & Temporal Convolutions)**: 
   利用多层不对称卷积核（如 $1 \times 2$、$4 \times 1$），模拟金融直觉中的“微观价格（Micro-price）”及订单失衡（Imbalance）特征计算。
3. **多尺度时序聚合 (Inception Module)**: 
   采用 Inception 模块并行不同大小的卷积核与最大池化层，等效于量化策略中配置不同半衰期衰减权重的多周期移动平均线（Moving Averages）。
4. **长程依靠近期 (LSTM / TCN)**: 
   通过 LSTM（或 TCN）解决金融时序中的长程因果推断与路径依赖问题。

---

## 🚀 实战量化架构优化 (Quant Architect Tweaks)

相比于原论文，本 PyTorch 实现拒绝了部分过于理想化的实验假设，为适应真实加密货币市场（Crypto Market）的高波与 Regime Switch 现象，进行了以下底层机制的重构：

### 1. 动态波动率标签生成 (Dynamic Threshold Labeling)
* **原论文**: 依赖静态的绝对收益率阈值 $\alpha$ 来划分涨（+1）、跌（-1）与平稳（0）区间。
* **本架构优化**: 引入基于滚动窗口的动态波动率自适应阈值：
  $$\text{DynamicThreshold}_t = \lambda \cdot \sigma_t$$
  其中 $\sigma_t$ 为基于过去 $W$ 个 tick 计算的标准差。此逻辑严格契合真实交易员基于当前波动率设定止盈止损的业务逻辑，有效应对波动率聚集（Volatility Clustering），在降噪的同时严防前瞻偏差（Lookahead Bias）。

### 2. 自适应流式归一化 (EWM Normalization)
* **原论文**: 采用回望过去 5 天数据的静态 Z-score 标准化。
* **本架构优化**: 使用具有 3600 秒半衰期的指数加权移动平均（Exponentially Weighted Moving Average, EWM）。该方案使模型在面对突发宏观冲击（如订单簿深度瞬间抽干）或市场微观状态切换时，能以极低的延迟自适应新的分布参数，提升系统的抗脆弱性。

### 3. 极度不平衡惩罚 (Focal Loss Integration)
* **原论文**: 采用标准的分类交叉熵损失（Categorical Cross-Entropy）。
* **本架构优化**: 高频 LOB 数据中往往存在超过 70% 的平稳（Stationary）样本。本代码移除了标准 CE，挂载了 `Focal Loss`：
  $$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  通过设置聚焦参数 $\gamma=2.0$，强制优化器将梯度下降的算力分配给那些因订单失衡导致微弱趋势的“难分样本（Hard Examples）”，极大地提升了趋势信号的召回率（Recall）。

### 4. 架构纵深与因果对比 (Architectural Capacity & TCN Baseline)
* 将特征感受野与通道容量进行了结构性扩容（输入窗口从 $T=100$ 延长至 $T=300$；特征通道数翻倍），并加入了 Dropout(0.2) 防止高维灾难。
* **因果对照组**: 额外实现并封装了具有膨胀因果卷积（Dilated Causal Convolutions）的 `TCN_PyTorch` 模型，作为严格防止未来信息泄露的 Benchmark 基线。

---

## ⚙️ 快速执行 (Quick Start)

环境依赖限定为量化标配：`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`。

执行流水线将自动完成“数据加载 -> 标签平滑与对齐 -> 流式归一化 -> DataLoader 构建 -> 滚动训练验证”：

```bash
python deepLOB.py