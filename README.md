# Dynamic Learned Index

## Running the code

```shell
# Setup the environment and install the dependencies
conda create -y -n DynamicLearnedIndex python=3.12
conda activate DynamicLearnedIndex
conda install -y -c pytorch faiss-cpu=1.8.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy loguru

# Dataset loading
pip install h5py

# BLISS
pip install scikit-learn

# Evaluation
pip install tqdm

# Download the datasets
wget 'https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'

# Run the code
python3 main.py
```
