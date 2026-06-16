import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import glob

WINDOW = 128
STEP   = 64
FS     = 50.0

LABEL_MAP = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}

def butter_lowpass(cutoff=0.3, fs=50.0, order=3):
    nyq = fs / 2
    return butter(order, cutoff / nyq, btype='low')

b, a = butter_lowpass()

def remove_gravity(total_acc):
    gravity = filtfilt(b, a, total_acc, axis=0)
    return total_acc - gravity

csv_files = glob.glob("../person_B/personal_data_*.csv")
if not csv_files:
    print("❌ 找不到 CSV 文件！")
    exit()

print(f"📂 找到 {len(csv_files)} 个文件：{csv_files}")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"✅ 总行数：{len(df)}")

df = df[df['label'].isin(LABEL_MAP.keys())].reset_index(drop=True)
print(f"✅ 有效标签行数：{len(df)}")
print(f"   各标签分布：\n{df['label'].value_counts()}")

total_acc = df[['ax', 'ay', 'az']].values
body_gyro = df[['gx', 'gy', 'gz']].values
body_acc  = remove_gravity(total_acc)

data_9ch = np.hstack([total_acc, body_acc, body_gyro])
labels   = df['label'].values

windows, window_labels = [], []

for start in range(0, len(data_9ch) - WINDOW, STEP):
    end = start + WINDOW
    segment   = data_9ch[start:end]
    seg_labels = labels[start:end]

    vals, counts = np.unique(seg_labels, return_counts=True)
    majority = vals[counts.argmax()]
    if counts.max() / WINDOW >= 0.8:
        windows.append(segment.T)
        window_labels.append(LABEL_MAP[majority])

X = np.array(windows)
y = np.array(window_labels)
print(f"\n✅ 总窗口数：{len(X)}")
print(f"   各类别分布：{np.unique(y, return_counts=True)}")

# 用个人数据自己的统计量归一化
mean = X.mean(axis=(0, 2), keepdims=True)  # (1, 9, 1)
std  = X.std(axis=(0, 2), keepdims=True)
np.save("personal_mean.npy", mean)
np.save("personal_std.npy", std)
print('After norm - mean:', ((X - mean)/(std+1e-8)).mean().round(4))
print('After norm - std:', ((X - mean)/(std+1e-8)).std().round(4))
X = (X - mean) / (std + 1e-8)

np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, y_train = X[idx[:split]],  y[idx[:split]]
X_test,  y_test  = X[idx[split:]], y[idx[split:]]

np.save("../person_B/X_train.npy", X_train)
np.save("../person_B/y_train.npy", y_train)
np.save("../person_B/X_test.npy",  X_test)
np.save("../person_B/y_test.npy",  y_test)
np.save("../person_B/personal_mean.npy", mean)
np.save("../person_B/personal_std.npy",  std)

print(f"\n✅ 已保存！训练集：{len(X_train)} 个，测试集：{len(X_test)} 个")
