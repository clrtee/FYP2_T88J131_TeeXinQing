# src/data_opportunity.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_opportunity(data_dir: str, window_len: int = 30, stride: int = 15):
    """
    Load raw Opportunity dataset from .dat files.
    - Features: 242 sensor columns (Index 1 to 242)
    - Label: Locomotion (Index 243 in raw 250-column .dat files)
    """
    X_all, y_all = [], []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Opportunity directory not found: {data_dir}. Please place .dat files here.")

    files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not files:
        raise FileNotFoundError(f"No .dat files found in {data_dir}")

    for f in sorted(files):
        path = os.path.join(data_dir, f)
        # 读入以空格分隔的 .dat 文件
        df = pd.read_csv(path, header=None, sep=r'\s+', engine="python")
        
        # 1. 提取特征: 第 2 列至第 243 列 (Index 1:243) 为 242 个原始传感器特征
        features = df.iloc[:, 1:243].astype(np.float32)
        
        # 处理原始数据中的大量 NaN (使用向前填充、向后填充，剩下的补 0)
        features = features.ffill().bfill().fillna(0)
        
        # 2. 提取 Locomotion 标签: 在原始文件中位于第 244 列 (Index 243)
        labels = df.iloc[:, 243].fillna(0).astype(int)
        
        # 将原始 Locomotion 标签映射到连续整数用于分类
        # 原始含义: 1(Stand), 2(Walk), 4(Sit), 5(Lie), 0(Null) -> 映射为 0, 1, 2, 3, 4
        label_map = {0: 4, 1: 0, 2: 1, 4: 2, 5: 3}
        labels = labels.map(label_map).fillna(4).astype(int)

        features_arr = features.values
        labels_arr = labels.values

        # 3. 滑动窗口切割 (Sliding Window)
        for start in range(0, len(features_arr) - window_len + 1, stride):
            end = start + window_len
            X_all.append(features_arr[start:end])
            # 取窗口内出现次数最多的标签作为整个窗口的标签
            y_all.append(np.bincount(labels_arr[start:end]).argmax())

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    # 按照时间序列 80/20 划分训练集和测试集
    n = len(X)
    cut = int(0.8 * n)
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]

    # 特征标准化 (Standardize)
    scaler = StandardScaler()
    Ntr, T, F = Xtr.shape
    Nte = Xte.shape[0]
    
    Xtr_2d = Xtr.reshape(Ntr * T, F)
    Xte_2d = Xte.reshape(Nte * T, F)
    
    Xtr_2d = scaler.fit_transform(Xtr_2d)
    Xte_2d = scaler.transform(Xte_2d)
    
    Xtr = Xtr_2d.reshape(Ntr, T, F)
    Xte = Xte_2d.reshape(Nte, T, F)

    return (Xtr, ytr), (Xte, yte)