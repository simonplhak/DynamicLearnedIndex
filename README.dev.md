## Compare with rust

You firstly need to run experiment that should match parameters of the Rust experiment. 
Then you can run the comparison with the following command in main branch:

```shell
python experiments/compare2rust_experiment.py
```

Then copy df_plot.csv content into `comparison/<current_date>.python_impl.csv`.

Then run experiment with rust implementation:

```shell
cargo build --release
./target/release/cli_dynamic_learned_index experiment compare2python  data/k300/ --skip-validation --start-from-one --force -n 1 5000 10000 30000 40000 50000 100000 --search-strategy model -i configs/python_vers.yaml
```

Then you can plot the results with:

```shell
python scripts/compare2python.py
```

Plot can be found in `comparison/<current_date>.rust2python.jpg`.

Experiments info

| Name | Note |
|------|------|
| 20250801 | initial comparison     |
| 20250802 | increase batch size for training     |
| 20250812 | usage of flat-knn lib inside a bucket      |
| 20250813 | usage of ncandidates       |