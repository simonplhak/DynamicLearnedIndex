## Compare with rust

You firstly need to run experiment that should match parameters of the Rust experiment. 
Then you can run the comparison with the following command in main branch:

```shell
python experiments/compare2rust_experiment.py
```

Then copy df_plot.csv content into `scripts/python_impl.csv`.

Then run experiment with rust implementation:

```shell
cargo build --release
./target/release/cli_dynamic_learned_index experiment compare2rust  data/k300/ --skip-validation --start-from-one --force -n 1 2 3 4 5 10 --search-strategy model -i configs/python_vers.yaml
```

Then you can plot the results with:

```shell
python scripts/compare2python.py
```

Plot can be found in [`plots/rust2python.jpg`](plots/rust2python.jpg).