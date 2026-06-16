# Privacy-Preserving Smart Home Using Federated Learning at the Edge

Final Year Project (FYP) — Faculty of Information Science and Technology (FIST), Multimedia University
Supervisor: Prof. Dr. Subarmaniam A/L Kannan

## Overview

This project investigates federated learning (FL) algorithm selection, differential privacy (DP) trade-offs, and edge hardware deployment for Human Activity Recognition (HAR) in a privacy-preserving smart home setting. The work follows a two-stage pipeline:

1. **Laptop-side FL simulation** — multi-client training (10 clients, 500 rounds) comparing FL algorithms under IID and non-IID (Dirichlet α = 0.1) data distributions, with and without client-side differential privacy.
2. **Edge deployment & personalised fine-tuning** — the pretrained global model is fine-tuned on a Raspberry Pi 4B using live IMU sensor data, simulating a real single-device adaptation scenario.

## Research Design

| Component | Details |
|---|---|
| Datasets | UCI-HAR, CASAS Aruba |
| Model architectures | 1D-CNN, LSTM |
| FL algorithms | FedAvg, FedProx, FedDC, SCAFFOLD |
| Data distribution | IID and non-IID (Dirichlet α = 0.1) |
| Privacy mechanism | DP-SGD (client-side, via Opacus) |
| Privacy budgets | ε ∈ {2, 4, 6, 8, 10} |
| Edge hardware | Raspberry Pi 4B + MPU6050 IMU sensor |
| FL framework | Flower (flwr) |

## Repository Structure

```
fyp_fl_smart_home/
├── src/                  # Core source code
│   ├── main.py           # Entry point for running experiments
│   ├── fl_client.py      # Flower client implementation
│   ├── fl_server.py      # Flower server / strategy orchestration
│   ├── models.py         # 1D-CNN and LSTM model definitions
│   ├── partition.py      # IID / non-IID (Dirichlet) data partitioning
│   ├── dp_utils.py        # DP-SGD / Opacus utilities
│   ├── strategy_scaffold.py
│   ├── strategy_feddc.py
│   ├── strategy_cdp.py
│   ├── scaffold_utils.py
│   ├── data_uci_har.py
│   ├── data_casas_aruba_csv.py
│   ├── centralized.py    # Centralized training benchmark
│   ├── device_profiles.py / profiler.py  # Resource profiling on edge hardware
│   └── train_utils.py / utils_history.py
│
├── hardware_exp/         # Raspberry Pi deployment & personalised fine-tuning
│   ├── pretrain/         # Pretrained global model artifacts
│   ├── rpi/ , rpi_clients/   # Edge-device client code
│   ├── person_A/ , person_B/ , person_C/  # Per-participant fine-tuning runs
│   ├── server.py
│   ├── finetune_results_dp.csv / finetune_results_nodp.csv
│   └── plot_*.png        # Hardware experiment plots
│
├── results/              # Raw experiment outputs (CSV) per dataset/model/algorithm/condition
├── notebooks/            # Analysis notebooks and result comparison plots
├── logs/                 # Server/client training logs per experiment configuration
├── models/               # Saved model checkpoints
├── data/                 # Dataset folders (raw data not included — see Datasets section)
├── run_*.sh              # Batch experiment shell scripts
└── requirements.txt
```

## Datasets

Raw datasets are **not included** in this repository due to size. Download them separately and place them under `data/`:

- **UCI-HAR**: [UCI Machine Learning Repository — Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **CASAS Aruba**: [WSU CASAS Smart Home Project](http://casas.wsu.edu/datasets/)

## Setup

```bash
git clone https://github.com/clrtee/fyp_fl_smart_home.git
cd fyp_fl_smart_home
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch, Flower (flwr) 1.7.0, and Opacus.

## Running Experiments

Experiments are launched via `src/main.py`; configuration (dataset, model, FL algorithm, IID/non-IID, DP parameters) is set via command-line arguments — run `python src/main.py --help` for the full argument list.

Batch experiment scripts are provided for reproducing the DP-SGD sweep across privacy budgets:

```bash
bash run_dp_fedprox_uci_batch64.sh
bash run_dp_fedprox_casas_batch64.sh
bash run_all_dp_fedprox_batch64.sh
```

Results are written as CSV files to `results/`, with corresponding training logs in `logs/`.

## Edge Deployment (Raspberry Pi)

After laptop-side FL simulation produces a pretrained global model (`models/`), the model is transferred to a Raspberry Pi 4B for personalised fine-tuning using live MPU6050 IMU data, comparing fine-tuning with and without DP-SGD. See `hardware_exp/` for the client/server scripts and per-participant results.

## Key Findings

- **FedProx** is recommended for edge deployment overall, due to lower communication cost and better noise stability than SCAFFOLD — though algorithm ranking is dataset-dependent (FedProx leads on UCI-HAR; SCAFFOLD leads on CASAS).
- **ε = 6** is the recommended privacy operating point for UCI-HAR (knee point between ε = 6–8 in the privacy–utility trade-off).
- CASAS Aruba suffers severe **minority class collapse** under DP-SGD (F1 drops ~39pp at ε = 2 while accuracy drops only ~20pp), making F1 the more revealing metric for imbalanced datasets.
- DP-induced accuracy degradation scales inversely with dataset size, and is markedly more severe in single-client hardware deployment than in multi-client simulation.
- LSTM + DP-SGD was excluded from the study due to Opacus hidden-state corruption causing model collapse.
- SCAFFOLD and FedDC require plain SGD (not Adam) at the client optimizer level, as Adam's adaptive scaling interferes with their theoretical gradient correction terms.

### Hardware Fine-Tuning Results (personalised fine-tuning, ε = 6, 5 rounds)

| Participant | Pretrained Accuracy | Fine-tuned (No DP) | Fine-tuned (DP) |
|---|---|---|---|
| Person A | 39.68% | 98.23% | 94.55% |
| Person B | 48.56% | 100% | 91.96% |
| Person C | 31.27% | 99.09% | 75.45% |

## Future Work

- Extend to multiple simultaneous edge clients for real-world FL scalability.
- Explore class-aware differential privacy to address minority class collapse in imbalanced datasets such as CASAS Aruba.

## Citation

If referencing this work, please cite it as:

```
Tee, X. Q. (2026). Privacy-Preserving Smart Home Using Federated Learning at the Edge.
Final Year Project, Faculty of Information Science and Technology, Multimedia University.
```
