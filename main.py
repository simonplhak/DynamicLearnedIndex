from __future__ import annotations

import socket
import time
from pathlib import Path

import torch
from faiss import METRIC_INNER_PRODUCT
from loguru import logger
from tqdm import tqdm

from configuration import DatasetConfig, DistanceConfig, ExperimentConfig, FrameworkConfig, SamplingConfig, SearchConfig
from leveling import Leveling
from lmi import LMIIndex
from plots import (
    plot_queries_per_second_vs_recall,
    plot_recall_vs_avg_time_per_query,
    plot_recall_vs_nprobe,
    save_relevant_results_to_csv,
)
from search_result import SearchResult
from utils import load_data, measure_runtime

SEED = 42
torch.manual_seed(SEED)

EXPERIMENTAL_RESULTS_DIR = Path('experimental_results')

experiment_id = time.strftime('%Y%m%d-%H%M%S')

logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'experiment.log', backtrace=True, diagnose=True)
logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'serialized.log', backtrace=True, diagnose=True, serialize=True)

if socket.gethostname() == 'Pro.local':
    experiment_config = ExperimentConfig(
        DatasetConfig(
            dataset_size=10_000,
            X=Path('laion2B-en-clip768v2-n=300K.h5'),
            Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
            GT=Path('gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        ),
        FrameworkConfig(
            LMIIndex,
            arity=3,
            bucket_shape=(200, 768),
            distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
            sampling=SamplingConfig(percentage=0.1, threshold=100_000),
        ),
        [SearchConfig(k=10, nprobe=nprobe) for nprobe in [1, 2]],
    )
else:
    experiment_config = ExperimentConfig(
        DatasetConfig(
            dataset_size=10_120_191,
            X=Path('laion2B-en-clip768v2-n=10M.h5'),
            Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
            GT=Path('gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        ),
        FrameworkConfig(
            LMIIndex,
            arity=3,
            bucket_shape=(3_000, 768),
            distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
            sampling=SamplingConfig(percentage=0.1, threshold=100_000),
        ),
        [SearchConfig(k=10, nprobe=nprobe) for nprobe in [1, 2, 3, 4, 5, 10, 25, 50, 100]],
    )

logger.info(f'Experiment ID: {experiment_id}')
logger.info(experiment_config)

X, Q, GT = load_data(experiment_config.dataset_config)

# Create the framework
framework = Leveling(experiment_config.framework_config)


# Insert the dataset one object at a time
@measure_runtime
def insert_objects(X: torch.Tensor) -> None:
    for i in range(len(X)):
        framework.insert(X[i], i)

        if (i + 1) % (len(X) // 10) == 0:
            logger.info(f'Inserted {i+1} objects')

        assert framework.get_n_objects() == i + 1, f'Wrong number of objects: {framework.get_n_objects()} != {i + 1}'

    logger.info(f'Inserted {len(X)} objects')


insert_objects(X)

# torch.save(framework, 'framework.pt')
# framework = torch.load('framework.pt')
framework.print_stats()
stats = framework.collect_stats()
logger.info(stats)


@measure_runtime
def perform_search(db_size: int, config: SearchConfig) -> SearchResult:
    recall_per_query = []
    n_candidates_per_query = []

    s = time.time()
    for i in tqdm(range(len(Q))):
        _, I, n_query_candidates = framework.search(Q[i], config.k, config.nprobe)
        recall = len(set((I[0] + 1).tolist()).intersection(set(GT[i, : config.k].tolist()))) / config.k

        recall_per_query.append(recall)
        n_candidates_per_query.append(n_query_candidates)
    e = time.time() - s

    return SearchResult(config, db_size, len(Q), recall_per_query, n_candidates_per_query, e)


# Search
results = []
for config in experiment_config.search_configs:
    logger.info(f'{config=}')
    result = perform_search(len(X), config)
    result.log_stats()
    results.append(result)
    # TODO: store result in a file

# Save results
experiment_dir = EXPERIMENTAL_RESULTS_DIR / experiment_id
experiment_dir.mkdir(exist_ok=True, parents=True)
(experiment_dir / 'experiment_id.txt').open('w').writelines([experiment_id])
(experiment_dir / 'experiment_config.txt').open('w').writelines([str(experiment_config)])
(experiment_dir / 'results.txt').open('w').writelines([str(results)])

# Save relevant plot data
df = save_relevant_results_to_csv(experiment_config, results, experiment_dir)

# Save plots
plot_recall_vs_nprobe(df, experiment_dir)
plot_queries_per_second_vs_recall(df, experiment_dir)
plot_recall_vs_avg_time_per_query(df, experiment_dir)
