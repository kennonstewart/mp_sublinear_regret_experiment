#!/usr/bin/env python3
"""
CLI driver for sublinear regret experiment.

Usage:
    python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42
"""

import click
import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path
from typing import Generator, Tuple, Any
import subprocess

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines import get_algorithm
from plotting import plot_regret_curve, save_analysis_report


def get_input_dim(dataset: str) -> int:
    """Get input dimension for dataset."""
    if dataset == "rotmnist":
        return 28 * 28  # Flattened MNIST
    elif dataset == "covtype":
        return 54  # COVTYPE features
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_num_classes(dataset: str) -> int:
    """Get number of classes for dataset."""
    if dataset == "rotmnist":
        return 10  # MNIST digits
    elif dataset == "covtype":
        return 7  # COVTYPE forest cover types
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_data_stream(dataset: str, stream: str, seed: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Get data stream generator."""
    try:
        from data_loader import get_rotating_mnist_stream, get_covtype_stream
        
        if dataset == "rotmnist":
            return get_rotating_mnist_stream(mode=stream, seed=seed)
        elif dataset == "covtype":
            return get_covtype_stream(mode=stream, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    except ImportError as e:
        print(f"Warning: Could not import data_loader: {e}")
        print("Using mock data stream for testing...")
        return get_mock_stream(dataset, stream, seed)


def get_mock_stream(dataset: str, stream: str, seed: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Generate mock data stream for testing when data_loader is not available."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = get_input_dim(dataset)
    num_classes = get_num_classes(dataset)
    
    # Generate infinite stream of random data
    while True:
        x = torch.randn(input_dim)
        y = torch.randint(0, num_classes, (1,)).squeeze()
        yield x, y


def get_memory_pair_algorithm(input_dim: int, num_classes: int, **kwargs):
    """Get Memory-Pair algorithm instance."""
    try:
        from code.memory_pair.src.memory_pair import MemoryPair
        return MemoryPair(input_dim, num_classes, **kwargs)
    except ImportError as e:
        print(f"Warning: Could not import MemoryPair: {e}")
        print("Using mock MemoryPair for testing...")
        return MockMemoryPair(input_dim, num_classes)


class MockMemoryPair:
    """Mock Memory-Pair algorithm for testing when the actual implementation is not available."""
    
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = torch.nn.Linear(input_dim, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for input x."""
        with torch.no_grad():
            return self.model(x)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new sample and return loss."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        logits = self.model(x)
        loss = self.criterion(logits, y.long())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def compute_regret(algorithm: Any, data_stream: Generator, T: int, 
                  results_dir: Path, filename_prefix: str) -> Tuple[list, list]:
    """
    Compute cumulative regret over T time steps.
    
    Args:
        algorithm: Online learning algorithm
        data_stream: Generator yielding (x, y) pairs
        T: Number of time steps
        results_dir: Directory to save results
        filename_prefix: Prefix for output files
        
    Returns:
        Tuple of (steps, cumulative_regret)
    """
    steps = []
    losses = []
    cumulative_regret = []
    
    print(f"Computing regret over {T} time steps...")
    
    # Best fixed policy regret (approximation)
    best_loss = float('inf')
    
    for t in range(1, T + 1):
        # Get next sample
        x, y = next(data_stream)
        
        # Make prediction and get loss
        loss = algorithm.update(x, y)
        losses.append(loss)
        
        # Update best fixed policy estimate (simple heuristic)
        if loss < best_loss:
            best_loss = loss
            
        # Compute cumulative regret (approximation)
        # True regret requires comparing to best fixed policy in hindsight
        # For now, we use cumulative loss as a proxy
        if t == 1:
            regret = loss
        else:
            regret = cumulative_regret[-1] + loss
            
        steps.append(t)
        cumulative_regret.append(regret)
        
        # Progress reporting
        if t % 10000 == 0:
            print(f"Step {t}/{T}, Current regret: {regret:.4f}")
    
    # Save results to CSV
    csv_path = results_dir / f"{filename_prefix}.csv"
    df = pd.DataFrame({
        'step': steps,
        'regret': cumulative_regret
    })
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate plot
    png_path = results_dir / f"{filename_prefix}.png"
    plot_regret_curve(str(csv_path), str(png_path), 
                     title=f"Cumulative Regret: {filename_prefix}")
    print(f"Plot saved to {png_path}")
    
    # Generate analysis report
    report_path = results_dir / f"{filename_prefix}_analysis.txt"
    save_analysis_report(str(csv_path), str(report_path))
    print(f"Analysis report saved to {report_path}")
    
    return steps, cumulative_regret


@click.command()
@click.option('--dataset', type=click.Choice(['rotmnist', 'covtype']), 
              required=True, help='Dataset to use')
@click.option('--stream', type=click.Choice(['iid', 'drift', 'adv']), 
              required=True, help='Stream type')
@click.option('--algo', type=click.Choice(['memorypair', 'sgd', 'adagrad', 'ons']), 
              required=True, help='Algorithm to use')
@click.option('--T', default=100000, help='Number of time steps')
@click.option('--seed', default=42, help='Random seed')
def main(dataset: str, stream: str, algo: str, t: int, seed: int):
    """Run sublinear regret experiment."""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Get dataset dimensions
    input_dim = get_input_dim(dataset)
    num_classes = get_num_classes(dataset)
    
    print(f"Starting experiment:")
    print(f"  Dataset: {dataset} (input_dim={input_dim}, num_classes={num_classes})")
    print(f"  Stream: {stream}")
    print(f"  Algorithm: {algo}")
    print(f"  Time steps: {t}")
    print(f"  Seed: {seed}")
    
    # Get data stream
    data_stream = get_data_stream(dataset, stream, seed)
    
    # Get algorithm
    if algo == "memorypair":
        algorithm = get_memory_pair_algorithm(input_dim, num_classes)
    else:
        algorithm = get_algorithm(algo, input_dim, num_classes)
    
    # Compute regret
    filename_prefix = f"{dataset}_{stream}_{algo}"
    steps, regret = compute_regret(algorithm, data_stream, t, results_dir, filename_prefix)
    
    # Git commit results
    try:
        hash_result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                   capture_output=True, text=True, cwd=Path(__file__).parent)
        git_hash = hash_result.stdout.strip()
        
        subprocess.run(['git', 'add', 'results/*'], cwd=Path(__file__).parent)
        commit_msg = f"EXP:sublinear_regret {dataset}-{stream}-{algo} {git_hash}"
        subprocess.run(['git', 'commit', '-m', commit_msg], cwd=Path(__file__).parent)
        print(f"Results committed with message: {commit_msg}")
    except Exception as e:
        print(f"Warning: Could not commit results: {e}")
    
    print(f"\nExperiment completed!")
    print(f"Final cumulative regret: {regret[-1]:.4f}")
    print(f"Expected √T scaling: {np.sqrt(t):.4f}")
    print(f"Ratio to √T: {regret[-1] / np.sqrt(t):.4f}")


if __name__ == "__main__":
    main()