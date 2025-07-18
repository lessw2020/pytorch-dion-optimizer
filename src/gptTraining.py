 """
Training Script: Dion vs AdamW Comparison

This script demonstrates the practical usage of the Dion optimizer and compares
its performance against AdamW on a language modeling task using a GPT-style transformer.

Key comparisons:
- Convergence speed (steps to target loss)
- Training stability
- Memory usage
- Wall-clock time performance
"""

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np

# Import our Dion optimizer
from dion_optimizer import DionOptimizer, create_dion_optimizer, get_parameter_info


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    # Model architecture
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 3072
    vocab_size: int = 50304
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # Training settings
    batch_size: int = 8
    max_steps: int = 3000
    eval_interval: int = 100
    eval_steps: int = 50
    
    # Optimizer settings
    dion_lr: float = 0.01
    dion_rank_factor: float = 0.25
    dion_momentum: float = 0.95
    dion_weight_decay: float = 0.01
    
    adamw_lr: float = 0.0002  # Typical AdamW LR for small models
    adamw_weight_decay: float = 0.01
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    
    # Data settings
    dataset_size: int = 100000
    synthetic_data: bool = True
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False
    seed: int = 42


class GPTModel(nn.Module):
    """GPT-style transformer model for language modeling."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm and language model head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {self.n_params/1e6:.1f}M")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        
        # Token and position embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        pos_emb = self.position_embedding(pos)      # (T, d_model)
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate query, key, value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(x)
        x = self.c_proj(x)
        return x


class SyntheticDataset(Dataset):
    """Synthetic dataset for language modeling."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_size = config.dataset_size
        
        # Generate random sequences
        self.data = torch.randint(
            0, config.vocab_size, 
            (config.dataset_size, config.max_seq_len + 1)
        )
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = sequence[:-1]
        targets = sequence[1:]
        return input_ids, targets


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    wall_time: float
    memory_used: float


class TrainingLogger:
    """Logger for training metrics and comparisons."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics: Dict[str, List[TrainingMetrics]] = {}
        self.start_time = time.time()
    
    def log_metrics(self, optimizer_name: str, metrics: TrainingMetrics):
        """Log metrics for a specific optimizer."""
        if optimizer_name not in self.metrics:
            self.metrics[optimizer_name] = []
        self.metrics[optimizer_name].append(metrics)
    
    def save_metrics(self, save_path: Path):
        """Save metrics to JSON file."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_metrics = {}
        for opt_name, metric_list in self.metrics.items():
            serializable_metrics[opt_name] = [asdict(m) for m in metric_list]
        
        with open(save_path / f"{self.experiment_name}_metrics.json", "w") as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def plot_comparison(self, save_path: Path):
        """Create comparison plots."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Optimizer Comparison: {self.experiment_name}", fontsize=16)
        
        colors = {'dion': 'blue', 'adamw': 'red'}
        
        for opt_name, metric_list in self.metrics.items():
            steps = [m.step for m in metric_list]
            losses = [m.loss for m in metric_list]
            wall_times = [m.wall_time for m in metric_list]
            grad_norms = [m.grad_norm for m in metric_list]
            memory_usage = [m.memory_used for m in metric_list]
            
            color = colors.get(opt_name, 'gray')
            
            # Loss vs steps
            axes[0, 0].plot(steps, losses, label=f'{opt_name}', color=color, alpha=0.8)
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss vs wall time
            axes[0, 1].plot(wall_times, losses, label=f'{opt_name}', color=color, alpha=0.8)
            axes[0, 1].set_xlabel('Wall Time (seconds)')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Loss vs Wall Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gradient norms
            axes[1, 0].plot(steps, grad_norms, label=f'{opt_name}', color=color, alpha=0.8)
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
            
            # Memory usage
            axes[1, 1].plot(steps, memory_usage, label=f'{opt_name}', color=color, alpha=0.8)
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Memory Usage (MB)')
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / f"{self.experiment_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        
        for opt_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
                
            final_loss = metric_list[-1].loss
            final_time = metric_list[-1].wall_time
            min_loss = min(m.loss for m in metric_list)
            
            # Find step where target loss was reached
            target_loss = min_loss * 1.05  # 5% above minimum
            steps_to_target = None
            time_to_target = None
            
            for m in metric_list:
                if m.loss <= target_loss:
                    steps_to_target = m.step
                    time_to_target = m.wall_time
                    break
            
            print(f"\n{opt_name.upper()}:")
            print(f"  Final Loss: {final_loss:.4f}")
            print(f"  Minimum Loss: {min_loss:.4f}")
            print(f"  Total Time: {final_time:.1f}s")
            if steps_to_target:
                print(f"  Steps to target: {steps_to_target}")
                print(f"  Time to target: {time_to_target:.1f}s")


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def compute_grad_norm(model):
    """Compute gradient norm for the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_with_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: TrainingConfig,
    optimizer_name: str,
    logger: TrainingLogger
) -> List[TrainingMetrics]:
    """Train model with specific optimizer and log metrics."""
    
    model.train()
    step = 0
    start_time = time.time()
    
    print(f"\nTraining with {optimizer_name.upper()}...")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    data_iter = iter(dataloader)
    
    while step < config.max_steps:
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, targets = next(data_iter)
        
        input_ids = input_ids.to(config.device)
        targets = targets.to(config.device)
        
        # Forward pass
        logits, loss = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm before clipping
        grad_norm = compute_grad_norm(model)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Log metrics
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            current_time = time.time() - start_time
            memory_used = get_memory_usage()
            
            metrics = TrainingMetrics(
                step=step,
                loss=loss.item(),
                learning_rate=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm,
                wall_time=current_time,
                memory_used=memory_used
            )
            
            logger.log_metrics(optimizer_name, metrics)
            
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Grad Norm: {grad_norm:.3f} | Time: {current_time:.1f}s")
        
        step += 1
    
    return logger.metrics[optimizer_name]


def run_comparison_experiment(config: TrainingConfig, save_dir: Path = Path("results")):
    """Run complete comparison experiment between Dion and AdamW."""
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Using device: {config.device}")
    print(f"Configuration: {config}")
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # Initialize logger
    logger = TrainingLogger(f"dion_vs_adamw_{config.d_model}M")
    
    # ========================================
    # Experiment 1: Train with Dion
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: DION OPTIMIZER")
    print(f"{'='*60}")
    
    # Create model
    model_dion = GPTModel(config).to(config.device)
    if config.compile_model and hasattr(torch, 'compile'):
        model_dion = torch.compile(model_dion)
    
    # Analyze parameters
    param_info = get_parameter_info(model_dion)
    print("\nParameter Analysis:")
    for key, value in param_info.items():
        print(f"  {key}: {value}")
    
    # Create Dion optimizer
    optimizer_dion = create_dion_optimizer(
        model_dion,
        lr=config.dion_lr,
        rank_factor=config.dion_rank_factor,
        momentum=config.dion_momentum,
        weight_decay=config.dion_weight_decay,
        scalar_optimizer='adamw'
    )
    
    print(f"Dion optimizer - Matrix params: {len(optimizer_dion.matrix_params)}, "
          f"Scalar params: {len(optimizer_dion.scalar_params)}")
    
    # Train with Dion
    train_with_optimizer(model_dion, optimizer_dion, dataloader, config, "dion", logger)
    
    # ========================================
    # Experiment 2: Train with AdamW
    # ========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: ADAMW OPTIMIZER")
    print(f"{'='*60}")
    
    # Create fresh model with same initialization
    torch.manual_seed(config.seed)  # Reset seed for identical initialization
    model_adamw = GPTModel(config).to(config.device)
    if config.compile_model and hasattr(torch, 'compile'):
        model_adamw = torch.compile(model_adamw)
    
    # Create AdamW optimizer
    optimizer_adamw = AdamW(
        model_adamw.parameters(),
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        weight_decay=config.adamw_weight_decay
    )
    
    # Train with AdamW
    train_with_optimizer(model_adamw, optimizer_adamw, dataloader, config, "adamw", logger)
    
    # ========================================
    # Results and Analysis
    # ========================================
    
    # Save results
    logger.save_metrics(save_dir)
    logger.plot_comparison(save_dir)
    logger.print_summary()
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS")
    print(f"{'='*60}")
    
    dion_metrics = logger.metrics["dion"]
    adamw_metrics = logger.metrics["adamw"]
    
    if dion_metrics and adamw_metrics:
        # Compare final performance
        dion_final_loss = dion_metrics[-1].loss
        adamw_final_loss = adamw_metrics[-1].loss
        
        dion_final_time = dion_metrics[-1].wall_time
        adamw_final_time = adamw_metrics[-1].wall_time
        
        print(f"\nFinal Loss Comparison:")
        print(f"  Dion: {dion_final_loss:.4f}")
        print(f"  AdamW: {adamw_final_loss:.4f}")
        print(f"  Improvement: {((adamw_final_loss - dion_final_loss) / adamw_final_loss * 100):+.1f}%")
        
        print(f"\nTraining Time Comparison:")
        print(f"  Dion: {dion_final_time:.1f}s")
        print(f"  AdamW: {adamw_final_time:.1f}s")
        print(f"  Speedup: {adamw_final_time / dion_final_time:.2f}x")
        
        # Memory usage comparison
        dion_peak_memory = max(m.memory_used for m in dion_metrics)
        adamw_peak_memory = max(m.memory_used for m in adamw_metrics)
        
        print(f"\nPeak Memory Usage:")
        print(f"  Dion: {dion_peak_memory:.1f} MB")
        print(f"  AdamW: {adamw_peak_memory:.1f} MB")
        print(f"  Difference: {dion_peak_memory - adamw_peak_memory:+.1f} MB")
    
    print(f"\nResults saved to: {save_dir}")
    print("Check the generated plots for detailed comparisons!")


def main():
    parser = argparse.ArgumentParser(description="Compare Dion vs AdamW optimizers")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=3000, help="Maximum training steps")
    parser.add_argument("--dion_lr", type=float, default=0.01, help="Dion learning rate")
    parser.add_argument("--adamw_lr", type=float, default=0.0002, help="AdamW learning rate")
    parser.add_argument("--rank_factor", type=float, default=0.25, help="Dion rank factor")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        dion_lr=args.dion_lr,
        adamw_lr=args.adamw_lr,
        dion_rank_factor=args.rank_factor,
        compile_model=args.compile
    )
    
    # Run experiment
    run_comparison_experiment(config, Path(args.save_dir))


if __name__ == "__main__":
    main()
