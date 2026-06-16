import os
import numpy as np
from sklearn.preprocessing import StandardScaler

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

def _load_txt(path: str) -> np.ndarray:
    return np.loadtxt(path)

def load_uci_har(uci_root: str):
    """
    UCI HAR "Inertial Signals" already segmented into fixed windows (128 timesteps).
    We'll use the 9 signals as features => [N, 128, 9].
    Labels are 1..6 => convert to 0..5.
    """
    train_dir = os.path.join(uci_root, "train")
    test_dir = os.path.join(uci_root, "test")

    def load_split(split_dir: str):
        inertial = os.path.join(split_dir, "Inertial Signals")
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        X_list = []
        for s in signals:
            fp = os.path.join(inertial, f"{s}_{os.path.basename(split_dir)}.txt")
            X_list.append(_load_txt(fp))  # [N, 128]
        # stack to [N, 128, 9]
        X = np.stack(X_list, axis=-1)
        y = _load_txt(os.path.join(split_dir, f"y_{os.path.basename(split_dir)}.txt")).astype(int) - 1
        return X, y

    Xtr, ytr = load_split(train_dir)
    Xte, yte = load_split(test_dir)

    # standardize features across dataset (flatten time)
    scaler = StandardScaler()
    X_all = np.concatenate([Xtr, Xte], axis=0)
    N, T, F = X_all.shape
    X_all_2d = X_all.reshape(N*T, F)
    X_all_2d = scaler.fit_transform(X_all_2d)
    X_all = X_all_2d.reshape(N, T, F)

    Xtr = X_all[: len(Xtr)]
    Xte = X_all[len(Xtr):]
    return (Xtr, ytr), (Xte, yte), scaler
