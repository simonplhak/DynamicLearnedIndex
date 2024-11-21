#!/bin/bash
#PBS -q elixircz@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=16:mem=1tb:scratch_local=200gb:cluster=elwe
#PBS -l walltime=24:00:00

export OMP_NUM_THREADS=$PBS_NUM_PPN

git config --global --add safe.directory /auto/brno12-cerit/nfs4/projects/fi-lmi-data/personal/david/research/DynamicLearnedIndex || exit 1

module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/prochazka/.conda/envs/DynamicLearnedIndex || exit 3

cd "$SCRATCHDIR" || exit 4
cp '/storage/brno12-cerit/home/prochazka/fi-lmi-data/data/LAION2B/laion2B-en-clip768v2-n=100M.h5' './laion2B-en-clip768v2-n=100M.h5' || exit 5

cd '/storage/brno12-cerit/home/prochazka/fi-lmi-data/personal/david/research/DynamicLearnedIndex' || exit 6
python3 main.py --scratch-folder="$SCRATCHDIR" &> '/storage/brno12-cerit/home/prochazka/fi-lmi-data/personal/david/research/DynamicLearnedIndex/metacentrum-logs/run.log'
CODE=$?

cd "$SCRATCHDIR" || exit 7
rm -r ./*
exit $CODE
