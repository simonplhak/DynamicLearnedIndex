#!/usr/bin/env python3
"""
Script to analyze logs.jsonl file and create visualizations of recall metrics vs nprobe values.
Creates 3 subplots showing recall_top1, recall_top5, and recall_top10 performance.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import click

def load_metrics_data(log_file_path):
    """
    Load metrics data from logs.jsonl file.
    
    Args:
        log_file_path (str): Path to the logs.jsonl file
        
    Returns:
        tuple: (metrics_data, test_dataset_size) where metrics_data is a list of dictionaries
               and test_dataset_size is the number of test queries
    """
    metrics_data = []
    test_dataset_size = None
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    # Filter lines with message: "metrics", "time" for function "eval_queries", or "dataset"
                    if data.get('message') == 'metrics':
                        metrics_data.append(data)
                    elif data.get('message') == 'time' and data.get('function') == 'eval_queries':
                        metrics_data.append(data)
                    elif data.get('message') == 'dataset' and 'test_dataset' in data:
                        test_dataset_size = data.get('test_dataset')
                        print(f"Found test dataset size: {test_dataset_size}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: File {log_file_path} not found")
        return [], None
    
    return metrics_data, test_dataset_size

def create_recall_plots(metrics_data, dataset_size, output_file=None):
    """
    Create 3 subplots showing recall metrics vs nprobe values.
    
    Args:
        metrics_data (list): List of metrics data dictionaries
        output_file (str, optional): Path to save the plot. If None, displays the plot.
    """
    if not metrics_data:
        print("No metrics data to plot")
        return
    
    # Extract data for plotting
    nprobes = []
    recall_top1 = []
    recall_top5 = []
    recall_top10 = []
    eval_times = []
    
    # Sort by nprobe value for plotting recall, and time for plotting eval times
    sorted_data_recall = sorted(metrics_data, key=lambda x: x.get('nprobe', 0))
    sorted_data_time = sorted(metrics_data, key=lambda x: x.get('timestamp', 0))

    for data in sorted_data_recall:
        if data.get('message') == 'metrics':
            nprobes.append(data.get('nprobe', 0))
            recall_top1.append(data.get('recall_top1', 0))
            recall_top5.append(data.get('recall_top5', 0))
            recall_top10.append(data.get('recall_top10', 0))

    for data in sorted_data_time:
        if data.get('message') == 'time' and data.get('function') == 'eval_queries':
            time_str = data.get('time', '0ms')
            # Parse time - handle both ms and s formats, convert to seconds
            if 'ms' in time_str:
                eval_times.append(float(time_str.replace('ms', '')) / dataset_size)
            elif 's' in time_str:
                eval_times.append(float(time_str.replace('s', '')) * 1000 / dataset_size)
            else:
                eval_times.append(float(time_str) / dataset_size)

    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Recall Metrics and Evaluation Time', fontsize=16)
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Plot 1: Recall Top-1
    ax1.plot(nprobes, recall_top1, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Probes (nprobe)')
    ax1.set_ylabel('Recall Top-1')
    ax1.set_title('Top-1 Recall')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(recall_top1) * 1.1)
    
    # Plot 2: Recall Top-5
    ax2.plot(nprobes, recall_top5, 'go-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Probes (nprobe)')
    ax2.set_ylabel('Recall Top-5')
    ax2.set_title('Top-5 Recall')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(recall_top5) * 1.1)
    
    # Plot 3: Recall Top-10
    ax3.plot(nprobes, recall_top10, 'ro-', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Probes (nprobe)')
    ax3.set_ylabel('Recall Top-10')
    ax3.set_title('Top-10 Recall')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(recall_top10) * 1.1)
    
    # Plot 4: Evaluation Time
    if eval_times:
        ax4.plot(nprobes, eval_times, 'mo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Evaluation Instance')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Evaluation Query Time')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No eval_queries time data found', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Evaluation Query Time (No Data)')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def print_summary_stats(metrics_data):
    """
    Print summary statistics of the metrics data.
    
    Args:
        metrics_data (list): List of metrics data dictionaries
    """
    if not metrics_data:
        return
    
    print("\n=== Metrics Summary ===")
    print(f"Total records: {len(metrics_data)}")
    
    nprobes = [data.get('nprobe', 0) for data in metrics_data]
    recall_top1 = [data.get('recall_top1', 0) for data in metrics_data]
    recall_top5 = [data.get('recall_top5', 0) for data in metrics_data]
    recall_top10 = [data.get('recall_top10', 0) for data in metrics_data]
    
    print(f"nprobe range: {min(nprobes)} - {max(nprobes)}")
    print(f"Recall Top-1 range: {min(recall_top1):.4f} - {max(recall_top1):.4f}")
    print(f"Recall Top-5 range: {min(recall_top5):.4f} - {max(recall_top5):.4f}")
    print(f"Recall Top-10 range: {min(recall_top10):.4f} - {max(recall_top10):.4f}")
    
    print("\nDetailed Results:")
    sorted_data = sorted(metrics_data, key=lambda x: x.get('nprobe', 0))
    for data in sorted_data:
        if data.get('message') == 'metrics':
            nprobe = data.get('nprobe', 0)
            top1 = data.get('recall_top1', 0)
            top5 = data.get('recall_top5', 0)
            top10 = data.get('recall_top10', 0)
            print(f"nprobe={nprobe:2d}: Top1={top1:.4f}, Top5={top5:.4f}, Top10={top10:.4f}")
        elif data.get('message') == 'time' and data.get('function') == 'eval_queries':
            eval_time = data.get('time')
            print(f"Eval Queries Time: {eval_time}")

@click.command()
@click.argument('log_file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='logs.jsonl')
@click.option('--output', '-o', type=str, help='Output file name for the plot (default: plot.nprobes.jpg)')
@click.option('--show-plot', is_flag=True, help='Display the plot instead of saving it')
def main(log_file, output, show_plot):
    """Analyze logs.jsonl file and create visualizations of recall metrics vs nprobe values.
    
    LOG_FILE: Path to the JSONL log file to analyze (default: logs.jsonl)
    """
    log_file_path = Path(log_file)
    
    # Load metrics data
    print(f"Loading metrics data from {log_file_path}...")
    metrics_data, test_dataset_size = load_metrics_data(str(log_file_path))
    
    if not metrics_data:
        print("No metrics data found")
        return
    
    if test_dataset_size is None:
        print("Warning: Test dataset size not found in logs. Using total time instead of per-query time.")
    
    # Print summary statistics
    print_summary_stats(metrics_data)
    
    # Determine output file or display option
    if show_plot:
        print("\nDisplaying plots...")
        create_recall_plots(metrics_data, test_dataset_size)
    else:
        # Create output filename
        if output:
            output_file = log_file_path.parent / output
        else:
            output_file = log_file_path.parent / "plot.nprobes.jpg"
        
        print(f"\nCreating and saving plots to {output_file}...")
        create_recall_plots(metrics_data, test_dataset_size, str(output_file))

if __name__ == "__main__":
    main()
