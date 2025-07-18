"""
Test script for the Dion optimizer.
This script creates a simple transformer model and trains it for 100 iterations
to verify that the loss decreases properly.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dion import create_dion_optimizer, DionOptimizer, SimpleTransformer


def test_dion_optimizer():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a simple transformer model
    model = SimpleTransformer(d_model=256, d_ff=1024, vocab_size=10000)

    # Create the Dion optimizer with AdamW as the scalar optimizer
    optimizer = create_dion_optimizer(
        model,
        lr=0.005,  # Increased learning rate
        rank_factor=0.25,
        scalar_optimizer="adamw",
        weight_decay=0.001,  # Reduced weight decay
        scalar_weight_decay=0.001,
        momentum=0.9,  # Adjusted momentum
    )

    # Training parameters
    batch_size = 16
    seq_len = 64
    num_iterations = 100
    vocab_size = 10000

    # Track losses
    losses = []

    # Create a synthetic task: predict next token based on current token
    # This gives the model a more structured task than random data
    print("Creating synthetic dataset...")
    data = torch.randint(0, vocab_size, (batch_size * 10, seq_len))

    print("Starting training loop...")
    for i in range(num_iterations):
        # Sample a batch
        batch_idx = torch.randint(0, data.size(0) - 1, (batch_size,))
        input_ids = data[batch_idx]

        # Target is to predict the next token (shifted by 1)
        target = torch.roll(input_ids, -1, dims=-1)
        # Mask the last position since we don't have a next token for it
        target[:, -1] = -100  # Ignore index for cross entropy

        # Forward pass
        embeddings = model.embedding(input_ids)

        # Apply attention (simplified)
        q = model.wq(embeddings)
        k = model.wk(embeddings)
        v = model.wv(embeddings)

        # Simple attention mechanism (not using actual transformer attention)
        attention = torch.matmul(q, k.transpose(-2, -1)) / (256**0.5)
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, v)

        # Output projection
        output = model.wo(context)
        output = model.norm1(output + embeddings)

        # Feed-forward network
        ff_output = model.ff2(torch.relu(model.ff1(output)))
        output = model.norm2(output + ff_output)

        # Language model head
        logits = model.lm_head(output)

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
        losses.append(current_loss)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {current_loss:.4f}")

    print("Training completed!")

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("dion_training_loss.png")
    print("Loss curve saved as 'dion_training_loss.png'")

    # Print loss statistics
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0]) * 100:.2f}%")

    return losses


if __name__ == "__main__":
    test_dion_optimizer()
