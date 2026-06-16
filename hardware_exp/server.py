import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
import sys
sys.path.append("..")
from src.models import CNN1DClassifier

model = CNN1DClassifier(input_dim=9, num_classes=6)
model.load_state_dict(torch.load("../pretrained_9ch.pth", map_location="cpu"))
initial_params = [val.cpu().numpy() for val in model.state_dict().values()]

strategy = fl.server.strategy.FedAvg(
    min_available_clients=1,
    min_fit_clients=1,
    min_evaluate_clients=1,
    initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
    fit_metrics_aggregation_fn=lambda metrics: {},
    evaluate_metrics_aggregation_fn=lambda metrics: {},
)

print("Server started, waiting for Pi...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
