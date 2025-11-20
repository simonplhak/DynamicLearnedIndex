# Simple project for translating pytorch models to candle rs

## Dev

To set up the development environment, follow these steps:

```bash
uv run sync
```

To generate pytorch model and test files, run:

```bash
uv run dump_pytorch_model.py
```

To run the Rust code that loads the model and test data, execute:

```bash
cargo run
```
