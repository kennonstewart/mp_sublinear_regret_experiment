"""
Plotting utilities for regret evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional


def plot_regret_curve(csv_path: str, output_path: str, 
                     title: Optional[str] = None, 
                     show_sqrt_guideline: bool = True):
    """
    Plot cumulative regret curve from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns (step, regret)
        output_path: Path to save the plot
        title: Optional title for the plot
        show_sqrt_guideline: Whether to show sqrt(T) guideline
    """
    # Read data
    df = pd.read_csv(csv_path)
    steps = df['step'].values
    regret = df['regret'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot regret curve
    ax.loglog(steps, regret, 'b-', linewidth=2, label='Cumulative Regret')
    
    # Add sqrt(T) guideline if requested
    if show_sqrt_guideline:
        sqrt_T = np.sqrt(steps)
        # Scale to match the regret curve magnitude
        scale_factor = regret[-1] / sqrt_T[-1] if sqrt_T[-1] > 0 else 1
        ax.loglog(steps, scale_factor * sqrt_T, 'r--', linewidth=1, 
                 label=f'√T guideline (scaled by {scale_factor:.2e})')
    
    # Formatting
    ax.set_xlabel('Time Step (T)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title(title or 'Cumulative Regret vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiple_regrets(csv_paths: List[str], labels: List[str], 
                         output_path: str, title: Optional[str] = None):
    """
    Plot multiple regret curves for comparison.
    
    Args:
        csv_paths: List of paths to CSV files
        labels: List of labels for each curve
        output_path: Path to save the plot
        title: Optional title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(csv_path)
        steps = df['step'].values
        regret = df['regret'].values
        
        color = colors[i % len(colors)]
        ax.loglog(steps, regret, color=color, linewidth=2, label=label)
    
    # Add sqrt(T) guideline
    if len(csv_paths) > 0:
        df = pd.read_csv(csv_paths[0])
        steps = df['step'].values
        sqrt_T = np.sqrt(steps)
        ax.loglog(steps, sqrt_T, 'k--', linewidth=1, alpha=0.5, label='√T guideline')
    
    # Formatting
    ax.set_xlabel('Time Step (T)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title(title or 'Cumulative Regret Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_regret_growth(csv_path: str) -> dict:
    """
    Analyze the growth rate of cumulative regret.
    
    Args:
        csv_path: Path to CSV file with columns (step, regret)
        
    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)
    steps = df['step'].values
    regret = df['regret'].values
    
    # Remove zeros and negative values for log analysis
    valid_mask = (steps > 0) & (regret > 0)
    steps_valid = steps[valid_mask]
    regret_valid = regret[valid_mask]
    
    if len(steps_valid) < 2:
        return {'error': 'Insufficient valid data points'}
    
    # Linear regression on log-log scale
    log_steps = np.log(steps_valid)
    log_regret = np.log(regret_valid)
    
    # Fit line: log(regret) = a * log(T) + b
    # This gives us regret = exp(b) * T^a
    coeffs = np.polyfit(log_steps, log_regret, 1)
    growth_exponent = coeffs[0]
    
    # Calculate final regret values
    final_regret = regret[-1]
    final_T = steps[-1]
    
    # Compare with sqrt(T) growth
    sqrt_T_final = np.sqrt(final_T)
    ratio_to_sqrt = final_regret / sqrt_T_final if sqrt_T_final > 0 else np.inf
    
    return {
        'growth_exponent': growth_exponent,
        'final_regret': final_regret,
        'final_T': final_T,
        'ratio_to_sqrt_T': ratio_to_sqrt,
        'is_sublinear': growth_exponent < 1.0,
        'is_sqrt_T_like': 0.4 <= growth_exponent <= 0.6  # Roughly sqrt(T) ~ T^0.5
    }


def save_analysis_report(csv_path: str, output_path: str):
    """
    Generate and save a text report of regret analysis.
    
    Args:
        csv_path: Path to CSV file with regret data
        output_path: Path to save the analysis report
    """
    analysis = analyze_regret_growth(csv_path)
    
    report = f"""
Regret Analysis Report
======================

Dataset: {Path(csv_path).stem}
Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
--------
Growth Exponent: {analysis.get('growth_exponent', 'N/A'):.4f}
Final Regret: {analysis.get('final_regret', 'N/A'):,.2f}
Final T: {analysis.get('final_T', 'N/A'):,}
Ratio to √T: {analysis.get('ratio_to_sqrt_T', 'N/A'):.4f}

Assessment:
-----------
Sublinear Growth: {'Yes' if analysis.get('is_sublinear', False) else 'No'}
√T-like Growth: {'Yes' if analysis.get('is_sqrt_T_like', False) else 'No'}

Interpretation:
---------------
"""
    
    if analysis.get('is_sublinear', False):
        report += "✓ The algorithm achieves sublinear regret (exponent < 1.0)\n"
    else:
        report += "✗ The algorithm does not achieve sublinear regret\n"
        
    if analysis.get('is_sqrt_T_like', False):
        report += "✓ The growth rate is consistent with √T scaling\n"
    else:
        report += "✗ The growth rate differs from √T scaling\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)