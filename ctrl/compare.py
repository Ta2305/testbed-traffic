#!/usr/bin/env python3
"""
WAFL Experiment Comparison Tool

Generates comparison graphs for multiple experiments:
1. Average Test Accuracy Curve (with SD band)
2. Cumulative Traffic Curve (line plot, no fill)

Experiments are ranked by total cumulative traffic (ascending) and
assigned colors accordingly.

Usage:
    python ctrl/compare.py exp1 exp2 exp3 ...

Example:
    python ctrl/compare.py Re_efficient_exp_0.1_8_128_2048-20260125T184236 \\
                           Re_efficient_exp_0.01_4_128_2048-20260124T235658
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_experiment_data(experiment_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load learning and network data from an experiment directory.
    
    Returns:
        Tuple of (learning_df, network_df)
    """
    all_learning = []
    all_network = []
    
    for device_id_str in os.listdir(experiment_path):
        device_path = os.path.join(experiment_path, device_id_str)
        if not (os.path.isdir(device_path) and device_id_str.isdigit()):
            continue
        
        # Load learning data
        learning_csv = os.path.join(device_path, "learning-data.csv")
        if os.path.exists(learning_csv):
            try:
                df = pd.read_csv(learning_csv)
                df["epoch"] = range(1, len(df) + 1)
                df["device_id"] = int(device_id_str)
                all_learning.append(df)
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load {learning_csv}: {e}")
        
        # Load network data
        network_csv = os.path.join(device_path, "network-data.csv")
        if os.path.exists(network_csv):
            try:
                df = pd.read_csv(network_csv)
                df["epoch"] = df["epoch"].astype(int)
                df["inbound_bytes"] = df["inbound_bytes"].astype(int)
                df["outbound_bytes"] = df["outbound_bytes"].astype(int)
                df["total_megabytes"] = (df["inbound_bytes"] + df["outbound_bytes"]) / 1e6
                df["device_id"] = int(device_id_str)
                all_network.append(df)
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load {network_csv}: {e}")
    
    learning_df = pd.concat(all_learning, ignore_index=True) if all_learning else pd.DataFrame()
    network_df = pd.concat(all_network, ignore_index=True) if all_network else pd.DataFrame()
    
    return learning_df, network_df


def calculate_total_cumulative_traffic(network_df: pd.DataFrame) -> float:
    """
    Calculate total cumulative traffic (sum of all devices at final epoch).
    """
    if network_df.empty:
        return float('inf')
    
    # Calculate cumulative traffic per device
    df = network_df.copy().sort_values(["device_id", "epoch"])
    df["cumulative_mb"] = df.groupby("device_id")["total_megabytes"].cumsum()
    
    # Get final epoch per device and sum
    final_traffic = df.groupby("device_id")["cumulative_mb"].max()
    return final_traffic.sum()


def prepare_accuracy_data(learning_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Prepare accuracy data for plotting (test accuracy only).
    """
    if learning_df.empty:
        return pd.DataFrame()
    
    # Extract test accuracy
    df = learning_df[["epoch", "device_id", "test_acc"]].copy()
    df = df.rename(columns={"test_acc": "accuracy"})
    df["experiment"] = experiment_name
    
    return df


def prepare_cumulative_traffic_data(network_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Prepare cumulative traffic data for plotting.
    """
    if network_df.empty:
        return pd.DataFrame()
    
    df = network_df.copy().sort_values(["device_id", "epoch"])
    df["cumulative_mb"] = df.groupby("device_id")["total_megabytes"].cumsum()
    
    # Calculate average across devices per epoch
    avg_df = df.groupby("epoch").agg({
        "cumulative_mb": ["mean", "std"]
    }).reset_index()
    avg_df.columns = ["epoch", "cumulative_mean", "cumulative_std"]
    avg_df["experiment"] = experiment_name
    
    # Also calculate total (sum of all devices)
    total_df = df.groupby("epoch")["cumulative_mb"].sum().reset_index()
    total_df = total_df.rename(columns={"cumulative_mb": "cumulative_total"})
    avg_df = avg_df.merge(total_df, on="epoch")
    
    return avg_df


def plot_accuracy_comparison(
    data: Dict[str, pd.DataFrame],
    rank_order: List[str],
    colors: Dict[str, str],
    output_path: str,
    self_epoch: int = 128,
    ylim_min: float = None
):
    """
    Plot accuracy comparison for multiple experiments.
    
    Args:
        ylim_min: If specified, set the y-axis minimum to this value
    """
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 14})
    
    # Combine all data
    all_data = []
    for exp_name in rank_order:
        if exp_name in data and not data[exp_name].empty:
            all_data.append(data[exp_name])
    
    if not all_data:
        print("❌ No accuracy data to plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Plot with seaborn
    ax = plt.gca()
    for i, exp_name in enumerate(rank_order):
        exp_df = combined_df[combined_df["experiment"] == exp_name]
        if exp_df.empty:
            continue
        
        label = f"#{i+1}: {exp_name}"
        color = colors[exp_name]
        
        # Calculate mean and std per epoch
        stats = exp_df.groupby("epoch")["accuracy"].agg(["mean", "std"]).reset_index()
        
        ax.plot(stats["epoch"], stats["mean"], label=label, color=color, linewidth=2)
        ax.fill_between(
            stats["epoch"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.2
        )
    
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Test Accuracy", fontsize=16)
    
    if ylim_min is not None:
        ax.set_ylim(bottom=ylim_min, top=1.0)
    ax.tick_params(axis='both', labelsize=14)
    
    # Add SELF/WAFL boundary line
    ax.axvline(self_epoch, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ylim = ax.get_ylim()
    ax.text(self_epoch, ylim[0] + (ylim[1] - ylim[0]) * 0.02, " → WAFL", 
            color="red", ha="left", va="bottom", fontsize=12)
    ax.text(self_epoch, ylim[0] + (ylim[1] - ylim[0]) * 0.02, "SELF ← ", 
            color="red", ha="right", va="bottom", fontsize=12)
    
    ax.legend(loc="lower right", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Saved accuracy comparison to: {output_path}")


def plot_traffic_comparison(
    data: Dict[str, pd.DataFrame],
    rank_order: List[str],
    colors: Dict[str, str],
    output_path: str,
    self_epoch: int = 128,
    log_scale: bool = False
):
    """
    Plot cumulative traffic comparison for multiple experiments.
    """
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 20})
    
    ax = plt.gca()
    
    for i, exp_name in enumerate(rank_order):
        if exp_name not in data or data[exp_name].empty:
            continue
        
        df = data[exp_name]
        label = f"#{i+1}: {exp_name}"
        color = colors[exp_name]
        
        # Plot total cumulative traffic (sum of all devices)
        ax.plot(df["epoch"], df["cumulative_total"], label=label, color=color, linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=16)
    ylabel = "Cumulative Traffic (MB, All Devices)"
    if log_scale:
        ax.set_yscale("log")
        ylabel += " [Log Scale]"
    ax.set_ylabel(ylabel, fontsize=16)
    
    ax.tick_params(axis='both', labelsize=14)
    
    # Add SELF/WAFL boundary line
    ax.axvline(self_epoch, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ylim = ax.get_ylim()
    if log_scale:
        import math
        y_pos = math.exp(math.log(ylim[0]) + (math.log(ylim[1]) - math.log(ylim[0])) * 0.02)
    else:
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.02
    ax.text(self_epoch, y_pos, " → WAFL", color="red", ha="left", va="bottom", fontsize=12)
    ax.text(self_epoch, y_pos, "SELF ← ", color="red", ha="right", va="bottom", fontsize=12)
    
    ax.legend(loc="upper left", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved traffic comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple WAFL experiments with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Names of experiment directories to compare (located in results/)"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: results)"
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name for the output subdirectory (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--self-epoch",
        type=int,
        default=128,
        help="Number of SELF learning epochs for boundary line (default: 128)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WAFL Experiment Comparison Tool")
    print("=" * 60)
    print(f"Experiments to compare: {len(args.experiments)}")
    
    # Validate experiments exist
    experiments = []
    traffic_totals = {}
    accuracy_data = {}
    traffic_data = {}
    final_accuracy = {}  # Store final epoch accuracy average for each experiment
    
    for exp_name in args.experiments:
        exp_path = os.path.join(args.results_dir, exp_name)
        if not os.path.isdir(exp_path):
            print(f"⚠️ Warning: Experiment not found: {exp_name}")
            continue
        
        print(f"\n📂 Loading: {exp_name}")
        learning_df, network_df = load_experiment_data(exp_path)
        
        if learning_df.empty and network_df.empty:
            print("  ❌ No data found, skipping")
            continue
        
        experiments.append(exp_name)
        
        # Calculate total traffic for ranking
        total_traffic = calculate_total_cumulative_traffic(network_df)
        traffic_totals[exp_name] = total_traffic
        print(f"  📡 Total cumulative traffic: {total_traffic:.2f} MB")
        
        # Prepare data for plotting
        accuracy_data[exp_name] = prepare_accuracy_data(learning_df, exp_name)
        traffic_data[exp_name] = prepare_cumulative_traffic_data(network_df, exp_name)
        
        # Calculate final accuracy average
        if not accuracy_data[exp_name].empty:
            final_epoch = accuracy_data[exp_name]["epoch"].max()
            final_acc = accuracy_data[exp_name][accuracy_data[exp_name]["epoch"] == final_epoch]["accuracy"].mean()
            final_accuracy[exp_name] = final_acc
            print(f"  📈 Final accuracy (avg): {final_acc:.4f}")
    
    if len(experiments) < 1:
        print("\n❌ No valid experiments to compare")
        sys.exit(1)
    
    # Rank experiments by traffic (descending)
    rank_order = sorted(experiments, key=lambda x: traffic_totals.get(x, float('inf')), reverse=True)
    
    print("\n" + "=" * 60)
    print("Experiment Rankings (by cumulative traffic, descending):")
    for i, exp_name in enumerate(rank_order):
        traffic = traffic_totals.get(exp_name, 0)
        print(f"  #{i+1}: {exp_name} ({traffic:.2f} MB)")
    print("=" * 60)
    
    # Assign colors
    n_experiments = len(rank_order)
    palette = sns.color_palette("tab10", n_colors=max(n_experiments, 10))
    colors = {exp_name: palette[i] for i, exp_name in enumerate(rank_order)}
    
    # Create output directory
    import datetime
    if args.output_name:
        output_subdir = args.output_name
    else:
        output_subdir = datetime.datetime.now().strftime("comparison_%Y%m%dT%H%M%S")
    
    output_dir = os.path.join(args.results_dir, "compare", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Generate plots
    print("\n📊 Generating comparison plots...")
    
    plot_accuracy_comparison(
        accuracy_data,
        rank_order,
        colors,
        os.path.join(output_dir, "accuracy_comparison.png"),
        self_epoch=args.self_epoch
    )
    
    plot_accuracy_comparison(
        accuracy_data,
        rank_order,
        colors,
        os.path.join(output_dir, "accuracy_comparison_0.8.png"),
        self_epoch=args.self_epoch,
        ylim_min=0.8
    )
    
    plot_traffic_comparison(
        traffic_data,
        rank_order,
        colors,
        os.path.join(output_dir, "traffic_comparison.png"),
        self_epoch=args.self_epoch,
        log_scale=False
    )
    
    plot_traffic_comparison(
        traffic_data,
        rank_order,
        colors,
        os.path.join(output_dir, "traffic_comparison_log.png"),
        self_epoch=args.self_epoch,
        log_scale=True
    )
    
    # Save ranking info
    ranking_path = os.path.join(output_dir, "ranking.txt")
    with open(ranking_path, "w") as f:
        f.write("Experiment Rankings (by cumulative traffic, ascending)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Experiment':<40} {'Traffic (MB)':<15} {'Accuracy':<10}\n")
        f.write("-" * 70 + "\n")
        for i, exp_name in enumerate(rank_order):
            traffic = traffic_totals.get(exp_name, 0)
            acc = final_accuracy.get(exp_name, 0)
            f.write(f"#{i+1:<5} {exp_name:<40} {traffic:>12.2f} MB  {acc:>8.4f}\n")
    print(f"  📝 Saved ranking info to: {ranking_path}")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
