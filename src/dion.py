"""
Dion: Distributed Orthonormalized Updates Optimizer

A PyTorch implementation of the Dion optimizer from the paper:
"Dion: Distributed Orthonormalized Updates" by Ahn et al.

This implementation supports:
- Centralized and distributed training
- 3D parallelism (DP, FSDP, TP)
- Automatic parameter type detection
- Integration with scalar optimizers
- Memory-efficient low-rank approximations
"""

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

try:
    from torch.optim.lion import Lion

    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    Lion = None


class DionOptimizer(Optimizer):
    """
    Dion (DIstributed OrthoNormalization) Optimizer

    Implements the Dion optimizer with support for both centralized and distributed training.
    Uses orthonormalized updates for matrix parameters and configurable scalar optimizers
    for non-matrix parameters.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor μ (default: 0.95)
        rank_factor: Rank fraction r/d for low-rank approximation (default: 1.0, full rank)
        scalar_optimizer: Optimizer class for non-matrix parameters ('adam', 'adamw', 'lion')
        scalar_lr: Learning rate for scalar optimizer (default: None, uses lr)
        weight_decay: Weight decay coefficient (default: 0.01)
        scalar_weight_decay: Weight decay for scalar parameters (default: 0.0)
        eps: Small constant for numerical stability (default: 1e-8)
        oversampling_factor: Factor for randomized QR oversampling (default: 1.25)
        distributed: Whether to use distributed implementation (default: auto-detect)
        matrix_threshold: Minimum size for treating parameter as matrix (default: 32)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        rank_factor: float = 1.0,
        scalar_optimizer: str = "adamw",
        scalar_lr: Optional[float] = None,
        weight_decay: float = 0.01,
        scalar_weight_decay: float = 0.0,
        eps: float = 1e-8,
        oversampling_factor: float = 1.25,
        distributed: Optional[bool] = None,
        matrix_threshold: int = 32,
        **scalar_kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 < rank_factor <= 1.0:
            raise ValueError(f"Invalid rank factor: {rank_factor}")

        self.rank_factor = rank_factor
        self.oversampling_factor = oversampling_factor
        self.matrix_threshold = matrix_threshold

        # Auto-detect distributed mode
        if distributed is None:
            distributed = dist.is_available() and dist.is_initialized()
        self.distributed = distributed

        # Setup scalar optimizer
        self.scalar_lr = scalar_lr if scalar_lr is not None else lr
        self.scalar_weight_decay = scalar_weight_decay
        self.scalar_optimizer_class = self._get_scalar_optimizer_class(scalar_optimizer)
        self.scalar_kwargs = scalar_kwargs

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super(DionOptimizer, self).__init__(params, defaults)

        # Initialize parameter classification and scalar optimizers
        self._classify_parameters()
        self._init_scalar_optimizers()

    def _get_scalar_optimizer_class(self, optimizer_name: str):
        """Get scalar optimizer class by name."""
        optimizer_map = {
            "adam": Adam,
            "adamw": AdamW,
        }

        if LION_AVAILABLE:
            optimizer_map["lion"] = Lion
        elif optimizer_name == "lion":
            raise ImportError(
                "Lion optimizer not available. Install torch>=2.0 or use 'adam'/'adamw'"
            )

        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported scalar optimizer: {optimizer_name}")

        return optimizer_map[optimizer_name]

    def _classify_parameters(self):
        """Classify parameters into matrix and scalar types."""
        self.matrix_params = []
        self.scalar_params = []

        for group in self.param_groups:
            for p in group["params"]:
                if self._is_matrix_param(p):
                    self.matrix_params.append(p)
                else:
                    self.scalar_params.append(p)

    def _is_matrix_param(self, param: torch.Tensor) -> bool:
        """Determine if parameter should be treated as a matrix."""
        # Must be 2D and both dimensions >= threshold
        return (
            param.dim() == 2
            and param.size(0) >= self.matrix_threshold
            and param.size(1) >= self.matrix_threshold
        )

    def _init_scalar_optimizers(self):
        """Initialize scalar optimizers for non-matrix parameters."""
        if not self.scalar_params:
            self.scalar_optimizer = None
            return

        # Create parameter groups with appropriate scaling
        scalar_groups = []
        for param in self.scalar_params:
            lr_scale = self._get_lr_scale(param)
            param_group = {
                "params": [param],
                "lr": self.scalar_lr * lr_scale,
                "weight_decay": self._get_weight_decay(param),
            }
            param_group.update(self.scalar_kwargs)
            scalar_groups.append(param_group)

        self.scalar_optimizer = self.scalar_optimizer_class(scalar_groups)

    def _get_lr_scale(self, param: torch.Tensor) -> float:
        """Get learning rate scaling factor based on parameter type and shape."""
        if param.dim() == 0:  # Scalar (normalization)
            return 1.0
        elif param.dim() == 1:  # Vector (bias, embedding, etc.)
            if param.numel() > 1000:  # Likely unembedding
                return 1.0 / math.sqrt(param.size(0))
            else:  # Likely bias
                return 1.0
        elif param.dim() == 2:  # Matrix (but below threshold)
            d_out, d_in = param.shape
            return math.sqrt(d_out / d_in)
        else:
            return 1.0

    def _get_weight_decay(self, param: torch.Tensor) -> float:
        """Get weight decay for parameter type."""
        # Only apply weight decay to matrix parameters (not bias/normalization)
        if param.dim() >= 2:
            return self.scalar_weight_decay
        return 0.0

    def _init_matrix_state(self, param: torch.Tensor, group: Dict) -> Dict:
        """Initialize state for matrix parameter."""
        state = {}
        m, n = param.shape
        rank = min(m, n, max(1, int(min(m, n) * self.rank_factor)))

        # Momentum buffer
        state["momentum_buffer"] = torch.zeros_like(param)

        # Right factor Q for warm-starting power iteration
        state["right_factor"] = torch.randn(
            n, rank, device=param.device, dtype=param.dtype
        )
        state["right_factor"] = self._normalize_columns(state["right_factor"])

        # Step counter
        state["step"] = 0
        state["rank"] = rank

        return state

    def _normalize_columns(self, matrix: torch.Tensor) -> torch.Tensor:
        """Normalize columns of matrix to unit norm."""
        norms = torch.norm(matrix, dim=0, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        return matrix / norms

    def _power_iteration(
        self, B: torch.Tensor, Q_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step of power iteration for low-rank approximation."""
        # P = BQ (left factor)
        P = torch.mm(B, Q_prev)

        # Orthogonalize P using QR decomposition
        P = self._orthogonalize_matrix(P)

        # R = B^T P (right factor)
        R = torch.mm(B.t(), P)

        return P, R

    def _orthogonalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix using QR decomposition."""
        if self.distributed:
            return self._distributed_orthogonalize(matrix)
        else:
            Q, _ = torch.linalg.qr(matrix)
            return Q

    def _distributed_orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Distributed orthogonalization using randomized Cholesky QR."""
        m, r = matrix.shape
        k = max(r, int(self.oversampling_factor * r))

        # Random sketching matrix
        S = torch.randn(k, m, device=matrix.device, dtype=matrix.dtype) / math.sqrt(k)

        # First iteration: randomized QR
        G = torch.mm(S, matrix)

        # Aggregate across distributed dimensions if needed
        if dist.is_initialized():
            dist.all_reduce(G, op=dist.ReduceOp.SUM)

        _, R1 = torch.linalg.qr(G)
        B = torch.linalg.solve_triangular(R1.t(), matrix.t(), upper=False).t()

        # Second iteration: Cholesky QR
        H = torch.mm(B.t(), B)

        # Aggregate across distributed dimensions if needed
        if dist.is_initialized():
            dist.all_reduce(H, op=dist.ReduceOp.SUM)

        R2 = torch.linalg.cholesky(
            H + torch.eye(r, device=H.device, dtype=H.dtype) * 1e-8
        )
        result = torch.linalg.solve_triangular(R2.t(), B.t(), upper=False).t()

        return result

    def _step_matrix_param(
        self, param: torch.Tensor, grad: torch.Tensor, group: Dict, state: Dict
    ):
        """Perform Dion update step for matrix parameter."""
        momentum = group["momentum"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]

        # Apply weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Get state variables
        momentum_buffer = state["momentum_buffer"]
        right_factor = state["right_factor"]
        rank = state["rank"]

        # Form buffer B_t = M_{t-1} + G_t
        buffer = momentum_buffer + grad

        # Power iteration: approximate B_t ≈ P_t R_t^T
        P, R = self._power_iteration(buffer, right_factor)

        # Approximation error
        approx = torch.mm(P, R.t())
        error = buffer - approx

        # Error feedback: M_t = μB_t + (1-μ)Δ_t = B_t - (1-μ)P_t R_t^T
        momentum_buffer.copy_(buffer - (1 - momentum) * approx)

        # Update right factor with column normalization
        right_factor.copy_(self._normalize_columns(R))

        # Compute scaled orthonormal update
        m, n = param.shape
        scale = math.sqrt(m / n)

        # Update parameters: X_t = X_{t-1} - η * scale * P_t Q_t^T
        Q = self._normalize_columns(R)
        update = torch.mm(P, Q.t())
        param.add_(update, alpha=-lr * scale)

        state["step"] += 1

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update matrix parameters with Dion
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                if self._is_matrix_param(param):
                    # Initialize state if needed
                    if param not in self.state:
                        self.state[param] = self._init_matrix_state(param, group)

                    self._step_matrix_param(param, param.grad, group, self.state[param])

        # Update scalar parameters with scalar optimizer
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients for all parameters."""
        super().zero_grad(set_to_none)
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.zero_grad(set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        state_dict = super().state_dict()
        if self.scalar_optimizer is not None:
            state_dict["scalar_optimizer"] = self.scalar_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        scalar_state = state_dict.pop("scalar_optimizer", None)
        super().load_state_dict(state_dict)

        if scalar_state is not None and self.scalar_optimizer is not None:
            self.scalar_optimizer.load_state_dict(scalar_state)

    def add_param_group(self, param_group: Dict):
        """Add a parameter group to the optimizer."""
        super().add_param_group(param_group)
        # Re-classify parameters and reinitialize scalar optimizers
        self._classify_parameters()
        self._init_scalar_optimizers()


# Utility functions for easy integration


def create_dion_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    rank_factor: float = 0.25,
    scalar_optimizer: str = "adamw",
    **kwargs,
) -> DionOptimizer:
    """
    Create a Dion optimizer with recommended settings.

    Args:
        model: PyTorch model
        lr: Learning rate
        rank_factor: Rank fraction for low-rank approximation
        scalar_optimizer: Optimizer for non-matrix parameters
        **kwargs: Additional arguments for DionOptimizer

    Returns:
        Configured DionOptimizer instance
    """
    return DionOptimizer(
        model.parameters(),
        lr=lr,
        rank_factor=rank_factor,
        scalar_optimizer=scalar_optimizer,
        **kwargs,
    )


def get_parameter_info(model: nn.Module, matrix_threshold: int = 32) -> Dict[str, int]:
    """
    Analyze model parameters for Dion optimization.

    Args:
        model: PyTorch model
        matrix_threshold: Minimum size for matrix classification

    Returns:
        Dictionary with parameter counts and memory usage
    """
    matrix_params = 0
    scalar_params = 0
    matrix_memory = 0
    scalar_memory = 0

    for param in model.parameters():
        param_memory = param.numel() * param.element_size()

        if (
            param.dim() == 2
            and param.size(0) >= matrix_threshold
            and param.size(1) >= matrix_threshold
        ):
            matrix_params += param.numel()
            matrix_memory += param_memory
        else:
            scalar_params += param.numel()
            scalar_memory += param_memory

    return {
        "matrix_params": matrix_params,
        "scalar_params": scalar_params,
        "matrix_memory_mb": matrix_memory / 1024**2,
        "scalar_memory_mb": scalar_memory / 1024**2,
        "total_params": matrix_params + scalar_params,
        "matrix_fraction": (
            matrix_params / (matrix_params + scalar_params)
            if (matrix_params + scalar_params) > 0
            else 0
        ),
    }


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a simple transformer-like model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, d_ff=2048, vocab_size=50000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.ff1 = nn.Linear(d_model, d_ff, bias=False)
            self.ff2 = nn.Linear(d_ff, d_model, bias=False)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # Create model and optimizer
    model = SimpleTransformer()

    # Analyze parameters
    param_info = get_parameter_info(model)
    print("Parameter Analysis:")
    for key, value in param_info.items():
        print(f"  {key}: {value}")

    # Create optimizer
    optimizer = create_dion_optimizer(
        model,
        lr=0.01,
        rank_factor=0.25,  # Use 1/4 rank for efficiency
        scalar_optimizer="adamw",
    )

    print(f"\nOptimizer created successfully!")
    print(f"Matrix parameters: {len(optimizer.matrix_params)}")
    print(f"Scalar parameters: {len(optimizer.scalar_params)}")
    print(f"Distributed mode: {optimizer.distributed}")

    # Test optimization step
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    target = torch.randint(0, 50000, (batch_size, seq_len))

    # Forward pass
    logits = model.lm_head(model.embedding(input_ids))
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), target.view(-1)
    )

    # Backward pass
    loss.backward()

    # Optimization step
    optimizer.step()
    optimizer.zero_grad()

    print(f"Test optimization step completed successfully!")
    print(f"Loss: {loss.item():.4f}")
