from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from analyze_metrics import load_metrics_data

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

metrics_data, test_dataset_size = load_metrics_data(Path(__file__).parent.parent / 'experiments_data' / 'compare2python' / 'logs.jsonl')
stats = defaultdict(dict)
df_rust = pd.DataFrame([metric for metric in metrics_data if metric['message'] == 'metrics'])
df_rust['avg_time_per_query'] = df_rust['elapsed_time'] / df_rust['total']
df_rust['avg_recall'] = df_rust['recall_top10'] * 100
df_rust['queries_per_second'] = 60 / df_rust['avg_time_per_query']
df_rust['type'] = 'rust'

        
df_python = pd.read_csv(Path(__file__).parent / 'python_impl.csv')
df_python['type'] = 'python'
df = pd.concat([df_python, df_rust])

plt.rcParams['figure.figsize'] = 30, 18
fig, axs = plt.subplots(2, 2, sharey=True)
axs = axs.flat

plot_recall_vs_nprobe(df, axs[0])
plot_queries_per_second_vs_recall(df, axs[1])
plot_recall_vs_avg_time_per_query(df, axs[2])

handles, labels = axs[2].get_legend_handles_labels()
fig.subplots_adjust(hspace=0.1, wspace=0.03) 
fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.3, 0.91))

plt.savefig(Path(__file__).parent.parent / 'plots' / 'rust2python.jpg')