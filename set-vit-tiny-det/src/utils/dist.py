"""
Distributed Training Utilities
Helper functions for multi-GPU training
"""

import torch
import torch.distributed as dist


def synchronize():
    """Synchronize between GPUs"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank():
    """Get current process rank"""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes"""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """Check if current process is main"""
    return get_rank() == 0