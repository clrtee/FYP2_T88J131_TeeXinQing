# src/data_casas_aruba_csv.py
import numpy as np
import pandas as pd


def load_casas_aruba_csv(
    csv_path: str,
    window_len: int = 50,
    stride: int = 25,
):
    """
    CASAS Aruba CSV (common format):
      date,time,sensor,value
    Example:
      2010-11-04,00:03:50.209589,Bedroom,ON

    Output:
      X: (N_windows, window_len, 2) -> [sensor_id_norm, value_bin]
      y: (N_windows,) pseudo-label (dominant sensor id in window)
    """

    # --- 1) Read with comma first ---
    df = pd.read_csv(csv_path, header=None, sep=",", engine="python")

    if df.shape[1] >= 4:
        # date, time, sensor, value
        df = df.iloc[:, :4].copy()
        df.columns = ["date", "time", "sensor", "value"]
        df = df.astype(str)
        df["timestamp"] = df["date"].str.strip() + " " + df["time"].str.strip()
        df = df[["timestamp", "sensor", "value"]]

    elif df.shape[1] == 3:
        # timestamp, sensor, value
        df = df.iloc[:, :3].copy()
        df.columns = ["timestamp", "sensor", "value"]

    else:
        # fallback whitespace split (rare)
        df = pd.read_csv(csv_path, header=None, sep=r"\s+", engine="python")
        if df.shape[1] >= 4:
            df = df.iloc[:, :4].copy()
            df.columns = ["date", "time", "sensor", "value"]
            df = df.astype(str)
            df["timestamp"] = df["date"].str.strip() + " " + df["time"].str.strip()
            df = df[["timestamp", "sensor", "value"]]
        elif df.shape[1] == 3:
            df = df.iloc[:, :3].copy()
            df.columns = ["timestamp", "sensor", "value"]
        else:
            raise ValueError(f"Cannot parse {csv_path}: got {df.shape[1]} cols, expected 3 or 4+")

    # --- clean ---
    df["sensor"] = df["sensor"].astype(str).str.strip()
    df["value"] = df["value"].astype(str).str.strip()

    df = df[df["sensor"].notna()]
    df = df[df["sensor"].str.len() > 0].reset_index(drop=True)

    # --- sensor mapping ---
    sensor_list = df["sensor"].unique().tolist()
    sensor2id = {s: i for i, s in enumerate(sensor_list)}
    df["sensor_id"] = df["sensor"].map(sensor2id).astype(int)

    num_sensors = len(sensor2id)
    if num_sensors > 5000:
        # Now this should never happen for Aruba if parsed correctly
        raise ValueError(
            f"num_sensors too large ({num_sensors}). "
            f"CSV parsing still wrong. Example row: {df.iloc[0].to_dict()}"
        )

    # --- value encoding ---
    v = df["value"].str.upper()
    value_bin = np.where(v.isin(["ON", "OPEN"]), 1.0,
                 np.where(v.isin(["OFF", "CLOSE"]), 0.0, 0.5)).astype(np.float32)

    sensor_id = df["sensor_id"].to_numpy(dtype=np.int64)
    denom = max(1, num_sensors - 1)
    sensor_id_norm = (sensor_id / denom).astype(np.float32)

    # --- sliding windows ---
    X, y = [], []
    n = len(df)

    for start in range(0, n - window_len + 1, stride):
        end = start + window_len

        x_win = np.stack([sensor_id_norm[start:end], value_bin[start:end]], axis=1)
        raw_ids = sensor_id[start:end]
        label = np.bincount(raw_ids).argmax()

        X.append(x_win)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, sensor2id, None


