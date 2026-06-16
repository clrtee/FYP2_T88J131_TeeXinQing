# src/strategy_scaffold.py

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
)
from flwr.server.client_proxy import ClientProxy

from .dp_utils import add_central_dp_noise, compute_updates
from .fl_client import pack_tensors, unpack_tensors
from .strategy_cdp import CentralDPStrategy

class ScaffoldStrategy(CentralDPStrategy):
    def __init__(self, cfg, **kwargs):
        """
        Scaffold Strategy 兼容 Central DP
        cfg: TrainConfig 实例，包含 use_central_dp, cdp_clip_norm, cdp_noise_multiplier 等
        """
        super().__init__(cfg=cfg, **kwargs) 
        self.cfg = cfg
        self.global_ctrl: Optional[List[np.ndarray]] = None  # 全局控制变量 c

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """下发全局控制变量 c_global 给客户端"""
        config: Dict[str, fl.common.Scalar] = {}
        
        # 获取默认的 FitIns（仅含模型参数）
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        # 将 c_global 注入每个客户端的配置中
        new_list = []
        for client, fit_ins in fit_ins_list:
            new_config = dict(fit_ins.config)
            new_config.update(config)
            new_list.append((client, fl.common.FitIns(fit_ins.parameters, new_config)))
        return new_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
    
        # 1. 提取客户端模型参数（用于聚合 delta_c）
        client_models = [parameters_to_ndarrays(res.parameters) for _, res in results]
    
        # 2. 调用父类聚合（FedAvg + DP 噪声注入全部在这里完成）
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result[0] is None:
            return None, {}
        agg_model_ndarrays = parameters_to_ndarrays(aggregated_result[0])
    
        # 3. 聚合控制变量增量 delta_c
        delta_c_blobs = []
        for _, res in results:
            if res.metrics and "delta_c" in res.metrics:
                delta_c_blobs.append(res.metrics["delta_c"])
    
        if delta_c_blobs:
            ref_tensors = [torch.from_numpy(p).cpu() for p in client_models[0]]
            delta_c_np_list = []
            for blob in delta_c_blobs:
                tensors = unpack_tensors(blob, torch.device("cpu"), ref_tensors)
                delta_c_np_list.append([t.detach().cpu().numpy() for t in tensors])
    
            mean_delta_c = []
            for layer_idx in range(len(delta_c_np_list[0])):
                layer_avg = np.mean([dc[layer_idx] for dc in delta_c_np_list], axis=0)
                mean_delta_c.append(layer_avg)
    
            if self.global_ctrl is None:
                self.global_ctrl = [np.zeros_like(arr) for arr in mean_delta_c]
            self.global_ctrl = [c + dc for c, dc in zip(self.global_ctrl, mean_delta_c)]
    
        # 4. 直接返回，DP 已由父类处理，不再重复注入
        return ndarrays_to_parameters(agg_model_ndarrays), aggregated_result[1]