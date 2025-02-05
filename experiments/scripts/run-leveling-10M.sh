#!/bin/bash
#PBS -q elixircz@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=8:mem=150gb:cluster=elwe
#PBS -l walltime=48:00:00
#PBS -o /storage/brno12-cerit/home/prochazka/projects/DynamicLearnedIndex/experiments/metacentrum-logs
#PBS -e /storage/brno12-cerit/home/prochazka/projects/DynamicLearnedIndex/experiments/metacentrum-logs
#PBS -m ae
#PBS -M davidprochazka@mail.muni.cz

export OMP_NUM_THREADS=$PBS_NUM_PPN

module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/prochazka/projects/DynamicLearnedIndex/env || exit 3

cd '/storage/brno12-cerit/home/prochazka/projects/DynamicLearnedIndex' || exit 4
python3 experiments/run.py \
    --compaction-strategy='leveling' \
    --shrink-buckets-during-compaction \
    --dataset-identifier='10M' \
    &>"/storage/brno12-cerit/home/prochazka/projects/DynamicLearnedIndex/experiments/metacentrum-logs/${PBS_JOBID}.LOG"
exit $?
