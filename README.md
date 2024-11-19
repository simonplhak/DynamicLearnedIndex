# Dynamic Learned Index

## Resources

- https://github.com/psu-db/dynamic-extension

## Running the code

```shell
# Setup the environment and install the dependencies
conda create -y -n DynamicLearnedIndex python=3.12
conda activate DynamicLearnedIndex
conda install -y -c pytorch faiss-cpu=1.9.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy loguru psutil

# Dataset loading
pip install h5py

# BLISS
pip install scikit-learn

# Evaluation
pip install tqdm

# Visualization
pip install seaborn pandas

# Download the datasets
wget 'https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'

# Run the code
python3 main.py
```
