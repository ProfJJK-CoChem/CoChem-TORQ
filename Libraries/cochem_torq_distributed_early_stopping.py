"""Multi-GPU Distributed Early Stopping and Deadlock Prevention Engine (REQ-TORQ-TRAIN-094 [D]).

Features:
- Count-weighted global validation loss reduction via torch.distributed.all_reduce(ReduceOp.SUM).
- Deadlock-free termination protocol with CPU-bound stop_flag broadcasting.
- Windows NT pickleable top-level worker functions and ephemeral dynamic port binding.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from Libraries.cochem_torq_training_errors import DistributedSyncError
from Libraries.cochem_torq_training_schemas import DistributedEarlyStoppingConfig


def find_free_port() -> int:
    """Acquire a free ephemeral port from the OS kernel for distributed rendezvous. [M]"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class DistributedEarlyStopping:
    """Stateful early stopping coordinator executing collective cross-rank reductions. [D]"""

    def __init__(self, config: DistributedEarlyStoppingConfig) -> None:
        self.config = config
        self.patience_epochs = config.patience_epochs
        self.min_delta = config.min_delta_hartree
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.early_stop: bool = False

    def step(
        self,
        local_loss_sum: float,
        local_count: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[bool, float]:
        """Process local shard metrics, reduce globally, and determine stopping condition. [D]
        
        Parameters
        ----------
        local_loss_sum : float
            Sum of validation losses across local shard batches.
        local_count : int
            Total number of samples in local shard.
        rank : int
            Rank index of current process.
        world_size : int
            Total number of ranks participating in DDP session.
            
        Returns
        -------
        Tuple[bool, float]
            (should_terminate, global_mean_validation_loss)
        """
        if world_size > 1 and dist.is_initialized():
            # Mandatory collective count-weighted aggregation across all ranks
            metric_tensor = torch.tensor(
                [float(local_loss_sum), float(local_count)],
                dtype=torch.float64,
                device="cpu",
            )
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

            total_loss_sum = metric_tensor[0].item()
            total_count = metric_tensor[1].item()

            if total_count <= 0.0:
                raise DistributedSyncError(
                    "Total sample count across distributed ranks evaluated to zero.",
                    diagnostics={"rank": rank, "world_size": world_size},
                )

            global_loss = total_loss_sum / total_count

            # Rank 0 evaluates patience against global loss
            stop_flag = torch.zeros(1, dtype=torch.int32, device="cpu")
            if rank == 0:
                if global_loss < self.best_loss - self.min_delta:
                    self.best_loss = global_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience_epochs:
                        self.early_stop = True
                        stop_flag.fill_(1)

            # Deadlock-Free Termination Protocol: CPU broadcast from rank 0
            dist.broadcast(stop_flag, src=0)
            should_terminate = bool(stop_flag.item() == 1)

            # Global barrier to synchronize ranks
            dist.barrier()
            return should_terminate, global_loss

        # Single-process fallback
        if local_count <= 0:
            raise DistributedSyncError("Local sample count evaluated to zero in early stopping.")

        global_loss = local_loss_sum / float(local_count)
        if global_loss < self.best_loss - self.min_delta:
            self.best_loss = global_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience_epochs:
                self.early_stop = True

        return self.early_stop, global_loss


def distributed_worker_routine(
    rank: int,
    world_size: int,
    port: int,
    shard_data: Dict[int, List[Tuple[float, int]]] | List[Tuple[float, int]],
    config_dict: Dict[str, Any],
    results_dict: Any,
) -> None:
    """Top-level pickleable worker function for multi-process distributed simulation. [M]"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # Initialize Gloo backend on CPU for cross-platform portability
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cfg = DistributedEarlyStoppingConfig(**config_dict)
        early_stopping = DistributedEarlyStopping(cfg)

        stopped_epoch = -1
        recorded_losses: List[float] = []

        if isinstance(shard_data, dict):
            epochs_data = shard_data[rank]
        else:
            epochs_data = shard_data

        for epoch, (loss_sum, count) in enumerate(epochs_data):
            should_stop, global_loss = early_stopping.step(
                local_loss_sum=loss_sum,
                local_count=count,
                rank=rank,
                world_size=world_size,
            )
            recorded_losses.append(global_loss)
            if should_stop:
                stopped_epoch = epoch
                break

        results_dict[rank] = {
            "stopped_epoch": stopped_epoch,
            "recorded_losses": recorded_losses,
            "final_best_loss": early_stopping.best_loss,
        }
    finally:
        dist.destroy_process_group()

