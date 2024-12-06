from __future__ import annotations

import pprint
import time
from pathlib import Path

import torch
from loguru import logger

import cli
from dynamic_learned_index import DynamicLearnedIndex
from execution_environment import detect_environment
from utils import (
    load_data,
    obtain_commit_hash,
    obtain_dirty_state,
)
from visualization.plots import (
    plot_queries_per_second_vs_recall,
    plot_recall_vs_avg_time_per_query,
    plot_recall_vs_nprobe,
    save_relevant_results_to_csv,
)

SEED = 42
torch.manual_seed(SEED)

EXPERIMENTAL_RESULTS_DIR = Path('experimental_results')

commit_hash, dirty_state = obtain_commit_hash(), obtain_dirty_state()
experiment_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{commit_hash}{'-dirty' if dirty_state else ''}"

logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'experiment.log', backtrace=True, diagnose=True)
logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'serialized.log', backtrace=True, diagnose=True, serialize=True)

args = cli.parse_arguments()
experiment_config = detect_environment().create_config(args, commit_hash, dirty_state)

logger.info(f'Experiment ID: {experiment_id}')
logger.info(experiment_config)

X, Q, GT = load_data(experiment_config.dataset_config)
logger.info(f'Loaded dataset of {len(X)} objects of size {(X.element_size() * X.nelement()) / 1024 ** 2:.0f} MB')

X = X.to(torch.float32)

# Create the index
dli = DynamicLearnedIndex(experiment_config.dli_config)
build_result = dli.insert_objects_sequentially(X)
dli.print_stats()

# Print stats once again
logger.info(f'Experiment ID: {experiment_id}')
logger.info(experiment_config)
logger.info(f'Build time: {build_result.time:.5}s')
logger.info(f'Insert throughput: {int(len(X)/build_result.time)} IPS')  # TODO: store persistently?
logger.info(f'Total model training time: {build_result.total_model_training_time():.3}s')
logger.info(f'Framework overhead time: {build_result.time - build_result.total_model_training_time():.3}s')
logger.info(f'When were the new levels allocated: {build_result.when_were_the_new_levels_allocated()}')
logger.info(f'Total number of retrained indexes: {build_result.total_n_retrained_indexes()}')
logger.info(pprint.pformat(build_result.stats))

# Save results
experiment_dir = EXPERIMENTAL_RESULTS_DIR / experiment_id
experiment_dir.mkdir(exist_ok=True, parents=True)
(experiment_dir / 'experiment_id.txt').open('w').writelines([experiment_id])
(experiment_dir / 'experiment_config.txt').open('w').writelines([pprint.pformat(experiment_config)])
(experiment_dir / 'build_result.txt').open('w').writelines([pprint.pformat(build_result)])

# Search
search_results = []
for config in experiment_config.search_configs:
    logger.info(config)
    result = dli.perform_search(len(X), config, Q, GT)
    logger.info(result.get_stats())
    logger.info(f'Search throughput: {int(len(Q)/result.total_search_time)} QPS')  # TODO: store persistently?

    # Save results
    (experiment_dir / 'search_results.txt').open('a').writelines([pprint.pformat(result), '\n'])

    search_results.append(result)

# Save relevant plot data
df = save_relevant_results_to_csv(experiment_config, build_result, search_results, experiment_dir)

# Save plots
plot_recall_vs_nprobe(df, experiment_dir)
plot_queries_per_second_vs_recall(df, experiment_dir)
plot_recall_vs_avg_time_per_query(df, experiment_dir)
