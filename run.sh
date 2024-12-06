#!/bin/bash
#PBS -q elixircz@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=8:mem=1tb:cluster=elwe
#PBS -l walltime=96:00:00

export OMP_NUM_THREADS=$PBS_NUM_PPN

git config --global --add safe.directory /auto/brno12-cerit/nfs4/projects/fi-lmi-data/personal/david/research/DynamicLearnedIndex || exit 1

module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/prochazka/.conda/envs/DynamicLearnedIndex || exit 3

cd '/storage/brno12-cerit/home/prochazka/fi-lmi-data/personal/david/research/DynamicLearnedIndex' || exit 4
python3 -OO main.py \
    --compaction-strategy='leveling' \
    --shrink-buckets-during-compaction='True' \
    &>"/storage/brno12-cerit/home/prochazka/fi-lmi-data/personal/david/research/DynamicLearnedIndex/metacentrum-logs/run-${PBS_JOBID}.log"
exit $?
