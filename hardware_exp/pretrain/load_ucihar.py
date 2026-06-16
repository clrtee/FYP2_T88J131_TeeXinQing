import numpy as np
import os

def load_ucihar(data_dir="UCI HAR Dataset"):
    def read_signals(folder, split):
        signals = []
        signal_names = [
            "total_acc_x", "total_acc_y", "total_acc_z",
            "body_acc_x",  "body_acc_y",  "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z"
        ]
        for name in signal_names:
            path = os.path.join(folder, split, "Inertial Signals", f"{name}_{split}.txt")
            signals.append(np.loadtxt(path))
        return np.stack(signals, axis=1)  # (N, 9, 128)

    def read_labels(folder, split):
        path = os.path.join(folder, split, f"y_{split}.txt")
        return np.loadtxt(path).astype(int) - 1  # 0-indexed

    X_train = read_signals(data_dir, "train")
    y_train = read_labels(data_dir, "train")
    X_test  = read_signals(data_dir, "test")
    y_test  = read_labels(data_dir, "test")

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2), keepdims=True)
    np.save("ucihar_mean.npy", mean)
    np.save("ucihar_std.npy",  std)
    print("✅ 已保存 ucihar_mean.npy 和 ucihar_std.npy")

    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    return X_train, y_train, X_test, y_test
