```markdown
# Privacy-Preserving Smart Home using Federated Learning (FYP)

This repository contains my Final Year Project implementation for **privacy-preserving smart home Human Activity Recognition (HAR)** using **Federated Learning (FL)** with an **LSTM** baseline model.  
It supports **non-IID client simulation (Dirichlet partitioning)**, multiple **FL algorithms**, **DP-SGD (Differential Privacy)**, and **edge device simulation** (latency / CPU / energy).

---

### Core ideas
- Train **HAR** models without sharing raw data (client-side training).
- Evaluate **FL algorithms** under **severe non-IID** distributions.
- Add **DP-SGD** to study the **privacy–utility trade-off**.
- Simulate deployment constraints of **heterogeneous edge devices**.

### Supported algorithms
- **FedAvg**
- **FedProx**
- **FedDC**
- **SCAFFOLD**

### Supported datasets
- **UCI-HAR** (smartphone accelerometer + gyroscope, 6 activities)
- **CASAS Aruba** (ambient smart home sensor events, multi-activity)

---

## Project structure

```

fyp_fl_smart_home/
├─ data/                  # datasets (you put/download here)
├─ models/                # saved models / checkpoints
├─ notebooks/             # analysis / plotting notebooks
├─ results/               # CSV logs, metrics, experiment outputs
├─ src/
│  ├─ main.py             # main entry (run experiments)
│  ├─ config.py           # experiment settings
│  ├─ data_uci_har.py     # UCI-HAR loader / processing
│  ├─ data_casas_aruba_csv.py  # CASAS Aruba loader / processing
│  ├─ partition.py        # IID / non-IID partition (Dirichlet)
│  ├─ models.py           # LSTM model definition
│  ├─ fl_server.py        # FL server coordination
│  ├─ fl_client.py        # FL client logic (local training)
│  ├─ train_utils.py      # training loop helpers
│  ├─ utils_history.py    # history / metrics logging
│  ├─ device_profiles.py  # device simulation profiles
│  ├─ profiler.py         # latency/energy/CPU simulation utils
│  ├─ strategy_feddc.py   # FedDC strategy
│  ├─ strategy_scaffold.py# SCAFFOLD strategy
│  └─ scaffold_utils.py   # SCAFFOLD helpers
└─ requirements.txt

````

---

## Requirements

- **Python 3.10+** (recommended: 3.12 as used in the project)
- Linux (tested on Linux Mint)

---

## Setup (recommended: virtual environment)

```bash
cd fyp_fl_smart_home

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
````

Check install:

```bash
python -c "import torch; print('torch:', torch.__version__)"
```

---

## Dataset setup

### Option A — Put datasets into `data/` (simple)

Create folders like:

```
data/
├─ uci_har/
└─ casas_aruba/
```

### UCI-HAR

1. Download the UCI-HAR dataset (Human Activity Recognition Using Smartphones).
2. Extract it into:

```
data/uci_har/
```

So it contains the typical `train/` and `test/` folders.

### CASAS Aruba

1. Prepare Aruba data as CSV (or the format your `data_casas_aruba_csv.py` expects).
2. Put it into:

```
data/casas_aruba/
```

> If your loader expects a specific filename (e.g., `aruba.csv`), keep the filename consistent with the code in `src/data_casas_aruba_csv.py`.

---

## How to run

### 1) Quick start (default config)

Most of the time, you can run using the settings in `src/config.py`:

```bash
python src/main.py
```

If you want to change experiments, edit:

* `src/config.py` (recommended)
  or
* parameters at the top of `src/main.py` (if your project is coded that way)

---

## Key experiment settings (where to change)

Open `src/config.py` and look for (names may be slightly different depending on your code, but these are the usual ones used in this repo):

### FL settings

* `DATASET`: `"uci_har"` or `"casas_aruba"`
* `ALGORITHM`: `"fedavg" | "fedprox" | "feddc" | "scaffold"`
* `NUM_CLIENTS`: e.g. `3`
* `NUM_ROUNDS`: e.g. `50`
* `LOCAL_EPOCHS`: e.g. `5`
* `BATCH_SIZE`: e.g. `64`
* `LEARNING_RATE`: e.g. `0.001`

### Non-IID partition (Dirichlet)

* `PARTITION_MODE`: `"iid"` or `"dirichlet"`
* `DIRICHLET_ALPHA`: e.g. `0.1` (severe non-IID), `0.01` (extreme non-IID)

### Differential Privacy (DP-SGD)

* `DP_ENABLED`: `True/False`
* `DP_EPSILON_LIST`: e.g. `[1.0, 5.0, 10.0]`
* `DP_DELTA`: e.g. `1e-5`
* `MAX_GRAD_NORM` (clip norm): e.g. `1.0`
* `NOISE_MULTIPLIER`: (if your code uses sigma/noise directly instead of epsilon)

### Edge device simulation

* `DEVICE_PROFILE`: e.g. `"rpi4" | "rpi5" | "jetson_nano" | "coral" | "esp32"`

Profiles are typically defined in `src/device_profiles.py`.

---

## Recommended experiment runs (matches your FYP hypotheses)

### Experiment 1 — Compare FL algorithms under non-IID

Goal: choose best algorithm under severe non-IID.

Suggested config:

* `DATASET = "uci_har"` (repeat for `"casas_aruba"`)
* `PARTITION_MODE = "dirichlet"`
* `DIRICHLET_ALPHA = 0.1`
* Run algorithms:

  * FedAvg, FedProx, FedDC, SCAFFOLD

Run:

```bash
python src/main.py
```

Repeat by changing `ALGORITHM` in `src/config.py`.

---

### Experiment 2 — Privacy–Utility trade-off (DP-SGD)

Goal: apply DP to the best FL algorithm and evaluate accuracy/F1 drop.

Suggested config:

* `ALGORITHM = "<best_from_exp1>"`
* `DP_ENABLED = True`
* `DP_EPSILON_LIST = [1.0, 5.0, 10.0]`
* Keep other settings fixed

Run:

```bash
python src/main.py
```

---

### Experiment 3 — Edge device simulation

Goal: simulate training cost / feasibility on heterogeneous edge devices.

Suggested config:

* `ALGORITHM = "<best_with_DP_or_without_DP>"`
* `DEVICE_PROFILE = "rpi4"` (repeat for rpi5, jetson_nano, coral, esp32)
* Collect:

  * latency per round
  * CPU usage %
  * energy (Wh)

Run:

```bash
python src/main.py
```

---

## Outputs (where results go)

Typical outputs:

* `results/`

  * CSV logs per run (round-by-round: loss / accuracy / f1 / latency / cpu / energy)
* `models/`

  * saved global model / checkpoints (if enabled)

If you don’t see outputs, check:

* `src/utils_history.py` (logging)
* `src/train_utils.py` (metrics)
* `src/main.py` (where files are saved)

---

## Common issues

### 1) `ModuleNotFoundError`

Make sure you installed dependencies:

```bash
pip install -r requirements.txt
```

### 2) Dataset not found / path error

Confirm your dataset folder names match what the loader expects:

* Check `src/data_uci_har.py`
* Check `src/data_casas_aruba_csv.py`

### 3) Non-IID partition looks “weird”

That’s normal with small client counts + small alpha.

* `alpha = 0.1` → severe non-IID
* `alpha = 0.01` → extreme non-IID (some clients may miss classes)

Partition logic lives in:

* `src/partition.py`

### 4) DP reduces accuracy too much

Try:

* larger `epsilon` (weaker privacy)
* smaller noise multiplier (if used)
* more rounds
* tune clip norm (`MAX_GRAD_NORM`)

---

## Reproducibility tips

* Fix random seeds (check `src/config.py` / `src/main.py`)
* Keep **same**: rounds, local epochs, batch size, learning rate
* Run each setting multiple times (e.g., 3–5 seeds) and report mean ± std

---

## Acknowledgements / libraries

* PyTorch (model training)
* Flower (federated learning framework)
* Opacus (DP-SGD integration)

---

## Author

**Tee Xin Qing**
Bachelor of Information Technology (Hons) Security Technology
Multimedia University


