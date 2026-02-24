# -*- coding: utf-8 -*-
"""
DeepLOB & TCN PyTorch 1:1 Parallel Replication
Role: Senior Quant AI Architect (Vibe Coding)
Source: main.ipynb | Target: PyTorch 2.6.0+cu124
"""

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn

# 设置设备 [对应 Cell 13]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子 [对应 Cell 13]
np.random.seed(1)
torch.manual_seed(2)

##############################################################################
# 1. 预处理与标签计算 (PREPROCESSING)
##############################################################################

# [Cell 4, 5] 加载数据与索引重排
def prepare_data(csv_path):
    # 修改前
    # df = pd.read_csv(csv_path, index_col='Unnamed: 0', parse_dates=True)
    # 修改后（假设日期格式为 '2023-01-09 22:17:40'）
    df = pd.read_csv(csv_path, index_col='Unnamed: 0', parse_dates=[2], date_format='%Y-%m-%d %H:%M:%S')
    print("-------------------------------")
    print(df.head())
    print("-------------------------------")

    df.columns = np.arange(42)
    df = df.drop_duplicates(subset=1)
    
    # 核心列重排逻辑: [AskP1, AskV1, BidP1, BidV1, ...]
    cols = [2,3,22,23,4,5,24,25,6,7,26,27,8,9,28,29,10,11,30,31,
            12,13,32,33,14,15,34,35,16,17,36,37,18,19,38,39,20,21,40,41]
    data = df.loc[:, cols]
    data.set_index(keys=pd.to_datetime(df[1]), drop=True, inplace=True)
    return data, df

# [Cell 7] 计算中间价
def get_midprice(df):
    midprice = pd.DataFrame((df.iloc[:,2] + df.iloc[:,22])/2, columns=['Price'])
    midprice['Time'] = pd.to_datetime(df[1])
    midprice.set_index(keys='Time', inplace=True)
    return midprice

# [Role: Quant Architect] 优化 1: 动态波动率标签生成
def labeling_dynamic(data, k, vol_window=100, lambda_factor=0.5):
    """
    基于动态波动率阈值的标签生成 (Dynamic Thresholding)
    :param vol_window: 计算波动率的回顾窗口 (如 100 ticks)
    :param lambda_factor: 波动率乘数，决定阈值的严格程度 (建议 0.5 ~ 1.0)
    """
    # 1. 计算未来平滑均值 (与原逻辑一致)
    data["MeanNegativeMid"] = data['Price'].rolling(window=k).mean()
    data["MeanPositiveMid"] = data["MeanNegativeMid"].shift(-(k-1))
    
    # 2. 计算未来变化率 l_t
    data["RateOfChange"] = (data["MeanPositiveMid"] - data['Price']) / data['Price']
    
    # 3. 计算动态阈值 (Dynamic Threshold)
    # 计算瞬时收益率
    returns = data['Price'].pct_change()
    # 计算滚动波动率 (标准差)
    data["Volatility"] = returns.rolling(window=vol_window).std()
    
    # 动态阈值 = lambda * 当前波动率
    # 注意：这里我们用当前的波动率来约束对未来的预测，这是合理的，因为交易员也是根据当前波动率设定止盈止损
    data["DynamicThreshold"] = data["Volatility"] * lambda_factor
    
    # 4. 生成标签
    # 初始化为 0 (Stationary)
    data[k] = 0
    
    # 过滤 NaN (前 vol_window 个数据无法计算波动率)
    valid_idx = data["DynamicThreshold"].dropna().index
    
    # Up: 变化率 > 动态阈值
    condition_up = data.loc[valid_idx, "RateOfChange"] > data.loc[valid_idx, "DynamicThreshold"]
    data.loc[valid_idx[condition_up], k] = 1
    
    # Down: 变化率 < -动态阈值
    condition_down = data.loc[valid_idx, "RateOfChange"] < -data.loc[valid_idx, "DynamicThreshold"]
    data.loc[valid_idx[condition_down], k] = -1
    
    # 清理中间列
    data = data.drop(columns=["MeanNegativeMid", "MeanPositiveMid", "RateOfChange", "DynamicThreshold", "Volatility"])
    
    return data.dropna()
    
# [Cell 9] 标签计算逻辑 (Method 1)
def labeling(data, k, alpha, type=1):
    data["MeanNegativeMid"] = data['Price'].rolling(window=k).mean()
    data["MeanPositiveMid"] = data["MeanNegativeMid"].shift(-(k-1))
    if type == 1:
        data["SmoothingLabel"] = (data["MeanPositiveMid"] - data['Price']) / data['Price']
    elif type == 2:
        data["SmoothingLabel"] = (data["MeanPositiveMid"] - data["MeanNegativeMid"]) / data["MeanNegativeMid"]
    
    labels_np = data["SmoothingLabel"].dropna()
    data[k] = None
    data.loc[labels_np.index, k] = 0 # Stationary
    data.loc[data["SmoothingLabel"] < -alpha, k] = -1 # Down
    data.loc[data["SmoothingLabel"] > alpha, k] = 1  # Up
    return data

##############################################################################
# 2. 归一化与数据加载 (DATASET & DATALOADER)
##############################################################################

# [Cell 14] DataSegmentation 类平行迁移
class LOBDataset(Dataset):
    def __init__(self, X, Y, window_size=300):
        # 对应 .reset_index(drop=True)
        self.X = torch.tensor(X.values, dtype=torch.float32)
        # Y 在 main.ipynb 中被 to_categorical 转换，这里直接存类别索引
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.window_size = window_size

    def __len__(self):
        # 对应 math.floor((len(self.X)-self.window_size)/ self.batch_size)
        # PyTorch 通过 DataLoader 处理 batch_size，这里返回总有效切片数
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # 对应 idx+=self.window_size -> x_sample=self.X.loc[idx-self.window_size:idx-1]
        x_sample = self.X[idx : idx + self.window_size]
        y_sample = self.Y[idx + self.window_size]
        # 增加 Channel 维以适配 Conv2D: (1, 300, 40)
        return x_sample.unsqueeze(0), y_sample

##############################################################################
# 3. DEEPLOB 模型构建 (MODEL ARCHITECTURE)
##############################################################################

# [Role: Quant Architect] 优化 3.1: Focal Loss 定义
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss 用于解决极度不平衡和难易样本问题
        :param gamma: 聚焦参数，越大越关注难分样本 (推荐 2.0)
        :param alpha: 类别权重 (Tensor), 类似 CrossEntropy 的 weight
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) logits, targets: (N) class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # 预测概率 p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

            
# [Cell 16] DeepLOB 原型平行迁移
class DeepLOB_PyTorch(nn.Module):
    def __init__(self, T=300, NF=40, number_of_lstm=64):
        super(DeepLOB_PyTorch, self).__init__()
        
        # 卷积块 1-3
        self.conv1 = nn.Conv2d(1, 32, (1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, (4, 1), padding='same')
        self.conv3 = nn.Conv2d(32, 32, (4, 1), padding='same')
        
        # 卷积块 4-6
        self.conv4 = nn.Conv2d(32, 32, (1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(32, 32, (4, 1), padding='same')
        self.conv6 = nn.Conv2d(32, 32, (4, 1), padding='same')
        
        # 卷积块 7-9
        self.conv7 = nn.Conv2d(32, 32, (1, 10))
        self.conv8 = nn.Conv2d(32, 32, (4, 1), padding='same')
        self.conv9 = nn.Conv2d(32, 32, (4, 1), padding='same')

        # Inception 模块
        self.inc1_1 = nn.Conv2d(32, 64, (1, 1), padding='same')
        self.inc1_2 = nn.Conv2d(64, 64, (3, 1), padding='same')
        self.inc2_1 = nn.Conv2d(32, 64, (1, 1), padding='same')
        self.inc2_2 = nn.Conv2d(64, 64, (5, 1), padding='same')
        self.inc3_1 = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))
        self.inc3_2 = nn.Conv2d(32, 64, (1, 1), padding='same')

        # LSTM 块 [Inception concat 后是 192 特征]
        self.lstm = nn.LSTM(192, number_of_lstm, batch_first=True)
        # 对应 keras.layers.Dropout(0.2, noise_shape=(None, 1, 192))
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(number_of_lstm, 3)

    def forward(self, x):
        # x: (N, 1, 300, 40)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.conv4(x), 0.01)
        x = F.leaky_relu(self.conv5(x), 0.01)
        x = F.leaky_relu(self.conv6(x), 0.01)
        x = F.leaky_relu(self.conv7(x), 0.01)
        x = F.leaky_relu(self.conv8(x), 0.01)
        x = F.leaky_relu(self.conv9(x), 0.01)

        # Inception
        x1 = F.leaky_relu(self.inc1_1(x), 0.01)
        x1 = F.leaky_relu(self.inc1_2(x1), 0.01)
        x2 = F.leaky_relu(self.inc2_1(x), 0.01)
        x2 = F.leaky_relu(self.inc2_2(x2), 0.01)
        x3 = F.leaky_relu(self.inc3_2(self.inc3_1(x)), 0.01)
        
        x = torch.cat([x1, x2, x3], dim=1) # (N, 192, 300, 1)
        x = x.squeeze(-1).transpose(1, 2) # (N, 300, 192)
        
        # 对应 Cell 16 的 Dropout 逻辑
        x = self.dropout(x)
        
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

##############################################################################
# 4. TCN 模型构建 (TEMPORAL CNN)
##############################################################################

# [Cell 79] TCN 残差块构建
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# [Cell 79] TCN 主体平行迁移
class TCN_PyTorch(nn.Module):
    def __init__(self, num_inputs=40, num_channels=[256]*6, kernel_size=2, dropout=0.2):
        super(TCN_PyTorch, self).__init__()
        layers = []
        # 对应 dilations=[1, 2, 4, 8, 16, 32]
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 3)

    def forward(self, x):
        # x: (N, 1, 300, 40) -> (N, 40, 300)
        x = x.squeeze(1).transpose(1, 2)
        y = self.network(x)
        return self.linear(y[:, :, -1])

##############################################################################
# 5. 训练、预测与绘图逻辑 (TRAIN, PREDICT, EVALUATE)
##############################################################################

# [Cell 32] 绘图逻辑平行迁移
def plot_results(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sn.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, cmap='RdYlGn', fmt='g')
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(y_true, y_pred, digits=4))
        
# [Role: Quant Architect] 优化 3.2: 训练循环挂载 Focal Loss        
# 通用训练循环 [对应 Cell 28,  fit]
def train_loop(model, train_loader, val_loader, checkpoint_path, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 替换标准 CrossEntropy
    # 如果某一类样本特别少，可以在这里计算 class_weights 并传入 alpha
    # 目前先单纯使用 Focal Loss 的 gamma 聚焦机制
    criterion = FocalLoss(gamma=2.0, alpha=None).to(device) 
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (out.argmax(1) == y).sum().item()
        
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_acc += (out.argmax(1) == y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc/len(val_loader.dataset):.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

# 预测与评估全过程 [对应 Cell 47, 81-83]
def run_evaluation(model, test_loader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y.numpy())
    
    plot_results(np.array(all_true), np.array(all_preds))

##############################################################################
# 6. 主执行流程 (MAIN PIPELINE)
##############################################################################

if __name__ == "__main__":
    # --- 1. 数据加载与预处理 (Cell 4-10) ---
    # 假设数据在 1-09-1-20.csv
    dataFile = './data/1-09-1-20.csv'
    data_raw, df_raw = prepare_data(dataFile)
    midprice = get_midprice(df_raw)
    # label_df = labeling(midprice, k=10, alpha=0.00001)
    label_df = labeling_dynamic(midprice, k=10, vol_window=100, lambda_factor=0.5)
    label_df.dropna(inplace=True)
    data_filtered = data_raw.loc[label_df.index]
    
    '''
    # --- 2. 归一化 (Cell 11) ---
    col_mean = data_filtered.rolling(86400).mean()
    col_std = data_filtered.rolling(86400).std()
    data_norm = (data_filtered - col_mean) / col_std
    data_norm.dropna(inplace=True)
    '''
    
    # --- 2. 归一化 --- 
    # [Role: Quant Architect] 优化 2: EWM 自适应归一化
    # 使用 3600 秒 (1小时) 的半衰期窗口，适应 Crypto 的快速 Regime Switch
    # 相比 rolling(86400)，这能让模型更快感知到波动率的变化
    col_mean = data_filtered.ewm(span=3600).mean()
    col_std = data_filtered.ewm(span=3600).std()
    
    # 防止除以 0 (极少数情况)
    col_std = col_std.replace(0, 1e-8)
    
    data_norm = (data_filtered - col_mean) / col_std
    data_norm.dropna(inplace=True)
    
    # --- 3. 标签对齐与编码 (Cell 12) ---
    final_labels = label_df.loc[data_norm.index, 10].values
    # 映射 -1, 0, 1 -> 0, 1, 2 (PyTorch Index 必须是非负)
    final_labels = (final_labels + 1).astype(int)

    # --- 4. 数据划分 (Cell 15) ---
    train_end = int(len(data_norm) * 0.6)
    val_end = train_end + int(len(data_norm) * 0.15)
    
    train_ds = LOBDataset(data_norm.iloc[:train_end], final_labels[:train_end])
    val_ds = LOBDataset(data_norm.iloc[train_end:val_end], final_labels[train_end:val_end])
    test_ds = LOBDataset(data_norm.iloc[val_end:], final_labels[val_end:])
    
    # 对应 batch_size=20 (Cell 15)
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=20, shuffle=False)

    # --- 5. 训练与评估 DeepLOB (Cell 18-77) ---
    print("\n--- Training DeepLOB Baseline ---")
    deeplob = DeepLOB_PyTorch(T=300, NF=40).to(device)
    train_loop(deeplob, train_loader, val_loader, 'deeplob_best.pth', epochs=10, lr=0.0001)
    run_evaluation(deeplob, test_loader, 'deeplob_best.pth')

    # --- 6. 训练与评估 TCN (Cell 78-83) ---
    print("\n--- Training TCN Baseline ---")
    tcn_model = TCN_PyTorch().to(device)
    train_loop(tcn_model, train_loader, val_loader, 'tcn_best.pth', epochs=10, lr=0.00001)
    run_evaluation(tcn_model, test_loader, 'tcn_best.pth')