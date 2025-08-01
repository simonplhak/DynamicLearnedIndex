from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from analyze_metrics import load_metrics_data
import datetime

def create_lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ax,
) -> None:
    """Create and save a line plot using seaborn.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: ax to plot

    """
    sns.lineplot(x=x, y=y, hue='type', data=df, ax=ax)
    ax.set_title(title)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    # Remove the legend for this specific axis
    ax.get_legend().remove() if ax.get_legend() else None

def plot_recall_vs_nprobe(df: pd.DataFrame, ax) -> None:
    create_lineplot(
        df=df,
        x='nprobe',
        y='avg_recall',
        title='Number of visited buckets vs. average recall',
        xlabel='Number of visited buckets',
        ylabel='Average recall',
        ax=ax,
    )


def plot_recall_vs_avg_time_per_query(df: pd.DataFrame, ax) -> None:
    create_lineplot(
        df=df,
        x='nprobe',
        y='avg_time_per_query',
        title='Number of visited buckets vs. average time per query',
        xlabel='Number of visited buckets',
        ylabel='Average time per query',
        ax=ax,
    )


def plot_queries_per_second_vs_recall(df: pd.DataFrame, ax) -> None:
    create_lineplot(
        df=df,
        x='avg_recall',
        y='queries_per_second',
        title='Average recall vs. queries per second',
        xlabel='Average recall',
        ylabel='Queries per second',
        ax=ax,
    )

def plot_build_index_time(df: pd.DataFrame, ax) -> None:
    sns.barplot(x='type', y='build_time', data=df, ax=ax)
    ax.set_title('Build time')  # type: ignore
    ax.set_xlabel('Implementation')  # type: ignore
    ax.set_ylabel('Build time')  # type: ignore
    # Remove the legend for this specific axis
    ax.get_legend().remove() if ax.get_legend() else None

current_date = datetime.datetime.now().strftime("%Y%m%d")
working_dir = Path(__file__).parent.parent / 'comparison'
metrics_data, test_dataset_size = load_metrics_data(Path(__file__).parent.parent / 'experiments_data' / 'compare2python' / 'logs.jsonl')
stats = defaultdict(dict)
df_rust = pd.DataFrame([metric for metric in metrics_data if metric['message'] == 'metrics'])
df_rust['avg_time_per_query'] = df_rust['elapsed_time'] / df_rust['total']
df_rust['avg_recall'] = df_rust['recall_top10'] * 100
df_rust['queries_per_second'] = 60 / df_rust['avg_time_per_query']
df_rust['type'] = 'rust'
build_time = None
for metric in metrics_data:
    if metric['message'] == 'time' and metric['function'] == 'insert_all_data':
         build_time = float(metric['time'].replace('s', ''))
assert build_time is not None
df_rust['build_time'] = build_time
df_rust.to_csv(working_dir / f'{current_date}.rust_impl.csv')

        
df_python = pd.read_csv(working_dir / '20250731.python_impl.csv')
df_python['type'] = 'python'
df = pd.concat([df_python, df_rust])

plt.rcParams['figure.figsize'] = 30, 18
fig, axs = plt.subplots(2, 2)
axs = axs.flat

plot_recall_vs_nprobe(df, axs[0])
plot_queries_per_second_vs_recall(df, axs[1])
plot_recall_vs_avg_time_per_query(df, axs[2])
plot_build_index_time(df, axs[3])

handles, labels = axs[2].get_legend_handles_labels()
fig.subplots_adjust(hspace=0.1, wspace=0.1) 
fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.3, 0.91))

plt.savefig(working_dir / f'{current_date}.rust2python.jpg')