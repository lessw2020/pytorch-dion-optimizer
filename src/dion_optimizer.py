"""
Dion Optimizer Module

This module exports the Dion optimizer classes and utility functions from dion_full_fsdp2.py
for use in training scripts like gptTraining.py.
"""

# Import and re-export the necessary components from dion_full_fsdp2.py
from dion_full_fsdp2 import (
    create_dion_optimizer,
    create_dion_optimizer_fsdp2,
    DionOptimizer,
    get_parameter_info,
    ShardingStrategy,
)

# Export all the necessary components
__all__ = [
    "DionOptimizer",
    "create_dion_optimizer",
    "create_dion_optimizer_fsdp2",
    "get_parameter_info",
    "ShardingStrategy",
]
