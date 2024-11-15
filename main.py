from __future__ import annotations

import pprint
import socket
import time
from pathlib import Path

import torch
from faiss import METRIC_INNER_PRODUCT
from loguru import logger
from tqdm import tqdm

from build_result import BuildResult
from configuration import DatasetConfig, DistanceConfig, ExperimentConfig, FrameworkConfig, SamplingConfig, SearchConfig
from experiment_search_result import ExperimentSearchResult
from leveling import Leveling
from lmi import LMIIndex
from plots import (
    plot_queries_per_second_vs_recall,
    plot_recall_vs_avg_time_per_query,
    plot_recall_vs_nprobe,
    save_relevant_results_to_csv,
)
from utils import load_data, measure_memory_usage, measure_runtime, obtain_commit_hash, obtain_dirty_state

SEED = 42
torch.manual_seed(SEED)

EXPERIMENTAL_RESULTS_DIR = Path('experimental_results')

commit_hash, dirty_state = obtain_commit_hash(), obtain_dirty_state()
experiment_id = f'{time.strftime('%Y%m%d-%H%M%S')}-{commit_hash}{'-dirty' if dirty_state else ''}'

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
        commit_hash,
        dirty_state,
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
        [SearchConfig(k=30, nprobe=nprobe) for nprobe in [1, 2, 3, 4, 5, 10, 25, 50, 100]],
        commit_hash,
        dirty_state,
    )

logger.info(f'Experiment ID: {experiment_id}')
logger.info(experiment_config)

X, Q, GT = load_data(experiment_config.dataset_config)

# Create the framework
framework = Leveling(experiment_config.framework_config)


# Insert the dataset one object at a time
@measure_runtime
@measure_memory_usage
def insert_objects(X: torch.Tensor) -> BuildResult:
    s = time.time()
    per_objects_insertion_statistics = []
    for i in range(len(X)):
        statistics = framework.insert(X[i], i)
        per_objects_insertion_statistics.append(statistics)

        if (i + 1) % (len(X) // 10) == 0:
            logger.info(f'Inserted {i+1} objects')

        assert framework.get_n_objects() == i + 1, f'Wrong number of objects: {framework.get_n_objects()} != {i + 1}'
    build_time = time.time() - s

    logger.info(f'Inserted {len(X)} objects')

    return BuildResult(build_time, framework.collect_stats(), per_objects_insertion_statistics)


build_result = insert_objects(X)
framework.print_stats()


@measure_runtime
def perform_search(db_size: int, config: SearchConfig) -> ExperimentSearchResult:
    recall_per_query = []
    n_candidates_per_query = []
    per_query_statistics = []

    s = time.time()
    for i in tqdm(range(len(Q))):
        _, I, statistics = framework.search(Q[i], config.k, config.nprobe)
        recall = len(set((I[0] + 1).tolist()).intersection(set(GT[i, : config.k].tolist()))) / config.k

        recall_per_query.append(recall)
        n_candidates_per_query.append(statistics.total_n_candidates)
        per_query_statistics.append(statistics)
    search_time = time.time() - s

    return ExperimentSearchResult(
        config,
        db_size,
        len(Q),
        recall_per_query,
        n_candidates_per_query,
        search_time,
        per_query_statistics,
    )


# Print stats once again
logger.info(f'Experiment ID: {experiment_id}')
logger.info(experiment_config)
logger.info(f'Build time: {build_result.time:.5}s')
logger.info(f'Insert throughput: {len(X)/build_result.time:.3} IPS')  # TODO: store persistently?
logger.info(f'Total model training time: {build_result.total_model_training_time():.3}s')
logger.info(f'Framework overhead time: {build_result.time - build_result.total_model_training_time():.3}s')
logger.info(pprint.pformat(build_result.stats))

# Search
search_results = []
for config in experiment_config.search_configs:
    logger.info(config)
    result = perform_search(len(X), config)
    logger.info(result.get_stats())
    logger.info(f'Search throughput: {len(Q)/result.total_search_time:.3} QPS')  # TODO: store persistently?
    search_results.append(result)

# Save results
experiment_dir = EXPERIMENTAL_RESULTS_DIR / experiment_id
experiment_dir.mkdir(exist_ok=True, parents=True)
(experiment_dir / 'experiment_id.txt').open('w').writelines([experiment_id])
(experiment_dir / 'experiment_config.txt').open('w').writelines([str(experiment_config)])
(experiment_dir / 'search_results.txt').open('w').writelines([str(search_results)])
(experiment_dir / 'build_result.txt').open('w').writelines([str(build_result)])

# Save relevant plot data
df = save_relevant_results_to_csv(experiment_config, build_result, search_results, experiment_dir)

# Save plots
plot_recall_vs_nprobe(df, experiment_dir)
plot_queries_per_second_vs_recall(df, experiment_dir)
plot_recall_vs_avg_time_per_query(df, experiment_dir)
