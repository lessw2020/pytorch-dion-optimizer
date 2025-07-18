"""
Comparison script for Dion optimizer vs AdamW optimizer.
This script creates two identical large transformer models and trains them on the same dataset,
one with Dion optimizer and one with AdamW optimizer, to compare their performance.
"""

import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dion import create_dion_optimizer


# Define a larger transformer model with multiple layers
class LargeTransformer(nn.Module):
    def __init__(self, d_model=2048, d_ff=8192, vocab_size=50000, num_layers=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Create multiple transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_ff) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        # Pass through each transformer layer
        for layer in self.layers:
            x = layer(x)

        # Language model head
        return self.lm_head(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Attention components
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Simple attention mechanism
        attention = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, v)

        # First residual connection and layer norm
        x = self.norm1(x + self.wo(context))

        # Feed-forward network
        ff_output = self.ff2(torch.relu(self.ff1(x)))

        # Second residual connection and layer norm
        x = self.norm2(x + ff_output)

        return x


def track_memory():
    """Track current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0  # Return 0 if CUDA is not available


def reset_memory_stats():
    """Reset memory tracking statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def compare_optimizers():
    # Set random seed for reproducibility
    torch.manual_seed(2020)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Common parameters for larger model
    d_model = 2048
    d_ff = 8192
    vocab_size = 50000
    num_layers = 32
    batch_size = 8  # Reduced batch size due to larger model
    seq_len = 64
    num_iterations = 100  # Reduced iterations for faster comparison
    lr = 0.001  # Reduced learning rate for stability with larger model
    weight_decay = 0.0001

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    data = torch.randint(0, vocab_size, (batch_size * 10, seq_len))
    if torch.cuda.is_available():
        data = data.to(device)

    # Results storage
    results = {
        "dion": {"losses": [], "time": 0, "peak_memory": 0},
        "adamw": {"losses": [], "time": 0, "peak_memory": 0},
    }

    # Test each optimizer
    for optimizer_name in ["dion", "adamw"]:
        print(f"\n{'='*50}")
        print(f"Testing {optimizer_name.upper()} optimizer")
        print(f"{'='*50}")

        # Create a new large model
        model = LargeTransformer(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        model.to(device)

        # Print model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model size: {total_params:,} parameters")

        # Create optimizer
        if optimizer_name == "dion":
            optimizer = create_dion_optimizer(
                model,
                lr=lr,
                rank_factor=0.25,
                scalar_optimizer="adamw",
                weight_decay=weight_decay,
                scalar_weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Reset memory stats
        reset_memory_stats()

        # Training loop
        start_time = time.time()
        print("Starting training loop...")

        for i in range(num_iterations):
            # Sample a batch
            batch_idx = torch.randint(0, data.size(0) - 1, (batch_size,))
            input_ids = data[batch_idx]

            # Target is to predict the next token (shifted by 1)
            target = torch.roll(input_ids, -1, dims=-1)
            # Mask the last position
            target[:, -1] = -100  # Ignore index for cross entropy

            # Forward pass using the model's forward method
            logits = model(input_ids)

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimization step
            optimizer.step()

            # Track loss
            current_loss = loss.item()
            results[optimizer_name]["losses"].append(current_loss)

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{num_iterations}, Loss: {current_loss:.4f}")

        # Record training time
        end_time = time.time()
        results[optimizer_name]["time"] = end_time - start_time

        # Record peak memory
        results[optimizer_name]["peak_memory"] = track_memory()

        print(f"Training completed!")
        print(f"Total time: {results[optimizer_name]['time']:.2f} seconds")
        print(f"Peak memory: {results[optimizer_name]['peak_memory']:.2f} MB")
        print(f"Initial loss: {results[optimizer_name]['losses'][0]:.4f}")
        print(f"Final loss: {results[optimizer_name]['losses'][-1]:.4f}")
        print(
            f"Loss reduction: {(1 - results[optimizer_name]['losses'][-1]/results[optimizer_name]['losses'][0]) * 100:.2f}%"
        )

    # Plot loss comparison
    plt.figure(figsize=(12, 6))
    plt.plot(results["dion"]["losses"], label="Dion")
    plt.plot(results["adamw"]["losses"], label="AdamW")
    plt.title("Training Loss Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("optimizer_comparison_loss.png")
    print("\nLoss comparison plot saved as 'optimizer_comparison_loss.png'")

    # Create bar charts for time and memory
    metrics = ["time", "peak_memory"]
    titles = ["Training Time (seconds)", "Peak Memory Usage (MB)"]
    filenames = ["optimizer_comparison_time.png", "optimizer_comparison_memory.png"]

    for metric, title, filename in zip(metrics, titles, filenames):
        plt.figure(figsize=(8, 6))
        values = [results["dion"][metric], results["adamw"][metric]]
        plt.bar(["Dion", "AdamW"], values)
        plt.title(title)
        plt.grid(axis="y")

        # Add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, f"{v:.2f}", ha="center")

        plt.savefig(filename)
        print(f"{title} comparison plot saved as '{filename}'")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<20} {'Dion':<15} {'AdamW':<15} {'Difference':<15}")
    print("-" * 65)

    # Loss reduction
    dion_loss_reduction = (
        1 - results["dion"]["losses"][-1] / results["dion"]["losses"][0]
    ) * 100
    adamw_loss_reduction = (
        1 - results["adamw"]["losses"][-1] / results["adamw"]["losses"][0]
    ) * 100
    loss_diff = dion_loss_reduction - adamw_loss_reduction
    print(
        f"{'Loss Reduction %':<20} {dion_loss_reduction:.2f}% {adamw_loss_reduction:.2f}% {loss_diff:+.2f}%"
    )

    # Training time
    time_diff_percent = (results["dion"]["time"] / results["adamw"]["time"] - 1) * 100
    print(
        f"{'Training Time':<20} {results['dion']['time']:.2f}s {results['adamw']['time']:.2f}s {time_diff_percent:+.2f}%"
    )

    # Memory usage
    if results["adamw"]["peak_memory"] > 0:  # Avoid division by zero
        memory_diff_percent = (
            results["dion"]["peak_memory"] / results["adamw"]["peak_memory"] - 1
        ) * 100
        print(
            f"{'Peak Memory':<20} {results['dion']['peak_memory']:.2f}MB {results['adamw']['peak_memory']:.2f}MB {memory_diff_percent:+.2f}%"
        )
    else:
        print(
            f"{'Peak Memory':<20} {results['dion']['peak_memory']:.2f}MB {results['adamw']['peak_memory']:.2f}MB N/A"
        )

    return results


if __name__ == "__main__":
    compare_optimizers()
