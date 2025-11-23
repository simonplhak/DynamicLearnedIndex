use std::{
    collections::HashMap,
    fs::File,
    io::{Read as _, Seek as _, SeekFrom, Write as _},
    path::Path,
};

use anyhow::Result;
use candle_core::{D, DType, Device, Result as CandleResult, Tensor, safetensors};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, loss, ops};
use clap::{Parser, Subcommand, command};
use rand::Rng as _;
use serde::{Deserialize, Serialize};

const DIM: usize = 3;
const HIDDEN_NEURONS: usize = 256;
const OUTPUT: usize = 10;

#[derive(Parser, Debug)]
#[command(name = "cli_app", version = "1.0", about = "CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    LoadPytorchModel,
    DumpModel,
    UseF16,
    DumpBuckets2Disk,
    LoadBucketsFromDisk,
}

enum CandleModelLayer {
    Linear(Linear),
    ReLU,
}
struct CandleModel {
    layers: Vec<CandleModelLayer>,
}

impl CandleModel {
    pub fn new(
        vb: VarBuilder,
        input_size: usize,
        layers: &[usize],
        output_size: usize,
    ) -> CandleResult<Self> {
        let mut current_input_size = input_size;
        let mut layers_list: Vec<CandleModelLayer> = Vec::new();

        for (i, &layer_size) in layers.iter().enumerate() {
            let prefix = format!("{i}", i = 2 * i);

            let linear_layer = candle_nn::linear(current_input_size, layer_size, vb.pp(&prefix))?;
            layers_list.push(CandleModelLayer::Linear(linear_layer));
            layers_list.push(CandleModelLayer::ReLU);

            current_input_size = layer_size;
        }

        let prefix = format!("{}", layers.len() * 2);
        let linear_layer_out = candle_nn::linear(current_input_size, output_size, vb.pp(&prefix))?;
        layers_list.push(CandleModelLayer::Linear(linear_layer_out));

        Ok(Self {
            layers: layers_list,
        })
    }

    pub fn train(&mut self, varmap: &VarMap, x: &Tensor, y: &Tensor) -> CandleResult<()> {
        let optim_config = candle_nn::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.0, // Make it behave like regular Adam
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(varmap.all_vars(), optim_config).unwrap();

        for _ in 0..3 {
            let logits = self.forward(x).unwrap();
            let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
            let loss = loss::nll(&log_sm, y).unwrap();
            opt.backward_step(&loss).unwrap();
        }
        Ok(())
    }
}

impl Module for CandleModel {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.layers
            .iter()
            .try_fold(xs.clone(), |acc, layer| match layer {
                CandleModelLayer::Linear(lin) => lin.forward(&acc),
                CandleModelLayer::ReLU => acc.relu(),
            })
    }
}

fn load_pytorch_model() -> Result<()> {
    // LOAD MODEL WEIGHTS //
    let weights_filename = Path::new("model.safetensors");
    let device = Device::Cpu;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };
    let model = CandleModel::new(vb, 3, &[256], 10)?;
    println!("Candle model loaded successfully");

    // LOAD TEST DATA //
    let test_path = Path::new("test_data.safetensors");
    let test_data: HashMap<String, Tensor> = safetensors::load(test_path, &device)?;

    let input_tensor = test_data.get("input").unwrap();
    let expected_output = test_data.get("output").unwrap();

    // RUN INFERENCE //
    println!("Running inference on Candle model...");
    let candle_output = model.forward(input_tensor)?;

    let diff = (&candle_output - expected_output)?.abs()?;
    println!(
        "Total difference: {}",
        diff.flatten(0, 1)?.sum(0)?.to_scalar::<f32>()?
    );
    Ok(())
}

fn dump_model() -> Result<()> {
    // CREATE MODEL //
    let varmap = VarMap::new();
    let device = Device::Cpu;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    println!("VarBuilder for f32 created successfully");
    let mut model = CandleModel::new(vb, DIM, &[HIDDEN_NEURONS], OUTPUT)?;

    // TRAIN MODEL TO INITIALIZE WEIGHTS //
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..10 * DIM).map(|_| rng.random_range(-1.0..1.0)).collect();
    let y: Vec<_> = (0..10)
        .map(|_| rng.random_range(0..OUTPUT as i64))
        .collect();
    let x = Tensor::from_slice(&x, &[10, DIM], &device)?;
    let y = Tensor::from_slice(&y, &[10], &device)?;
    model.train(&varmap, &x, &y)?;

    // SAVE MODEL WEIGHTS //
    let weights_filename = Path::new("rust-model.safetensors");
    varmap.save(weights_filename)?;
    println!("Model weights dumped to {weights_filename:?}");

    // RUN INFERENCE //
    let test_x: Vec<f32> = (0..5 * DIM).map(|_| rng.random_range(-1.0..1.0)).collect();
    let test_x = Tensor::from_slice(&test_x, &[5, DIM], &device)?;
    let test_y = model.forward(&test_x)?;

    // SAVE TEST DATA //
    let mut tensors = HashMap::new();
    tensors.insert("input".to_string(), test_x);
    tensors.insert("output".to_string(), test_y);
    let test_data_filename = Path::new("rust-test_data.safetensors");
    safetensors::save(&tensors, test_data_filename)?;
    println!("Test data dumped to {test_data_filename:?}");
    Ok(())
}

fn use_f16() -> Result<()> {
    // LOAD MODEL WEIGHTS //
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
    println!("VarBuilder for f16 created successfully");
    let model = CandleModel::new(vb, DIM, &[HIDDEN_NEURONS], OUTPUT)?;
    println!("Candle model with f16 loaded successfully");

    // LOAD TEST DATA //
    let test_path = Path::new("test_data.safetensors");
    let test_data: HashMap<String, Tensor> = safetensors::load(test_path, &device)?;

    let input_tensor = test_data.get("input").unwrap().to_dtype(DType::F16)?;

    // RUN INFERENCE //
    println!("Running inference on Candle model with f16...");
    let candle_output = model.forward(&input_tensor)?;
    println!("output: {candle_output}");
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct DiskBucket {
    offset: u64,
    count: usize,
}

fn dump_vector2disk() -> Result<()> {
    // SIMULATE BUCKETS //
    let bucket_1 = (0..3).flat_map(|i| vec![i as f32; DIM]).collect::<Vec<_>>();
    let bucket_2 = (3..5).flat_map(|i| vec![i as f32; DIM]).collect::<Vec<_>>();
    let bucket_3 = (5..6).flat_map(|i| vec![i as f32; DIM]).collect::<Vec<_>>();
    let buckets = [bucket_1, bucket_2, bucket_3];
    println!("Buckets: {buckets:?}");

    // DUMP TO DISK //
    let mut file = File::create("buckets.bin")?;
    let mut buckets_meta = Vec::new();
    for bucket in buckets.iter() {
        let offset = file.stream_position()?;
        let bytes: &[u8] = bytemuck::cast_slice(bucket);
        file.write_all(bytes)?;
        buckets_meta.push(DiskBucket {
            offset,
            count: bucket.len(),
        });
    }
    println!("Buckets dumped to buckets.bin");

    // DUMP META TO DISK //
    let meta_filename = Path::new("buckets_meta.json");
    let meta_file = File::create(meta_filename)?;
    serde_json::to_writer(meta_file, &buckets_meta)?;
    println!("Buckets meta dumped to {meta_filename:?}");
    Ok(())
}

fn load_vector_from_disk() -> Result<()> {
    // LOAD META FROM DISK //
    let meta_filename = Path::new("buckets_meta.json");
    let meta_file = File::open(meta_filename)?;
    let buckets_meta: Vec<DiskBucket> = serde_json::from_reader(meta_file)?;
    println!("Loaded buckets meta: {buckets_meta:?}");

    // LOAD BUCKETS FROM DISK //
    let mut file = File::open("buckets.bin")?;
    let dtype_size = std::mem::size_of::<f32>();
    let mut buckets = Vec::new();
    for bucket_meta in buckets_meta.iter() {
        let start = bucket_meta.offset as usize;
        let end = start + bucket_meta.count * dtype_size;
        let mut buffer = vec![0u8; end - start];
        // Seek here is unnecessary since we read sequentially, but included for clarity
        file.seek(SeekFrom::Start(bucket_meta.offset))?;
        file.read_exact(&mut buffer)?;
        let bucket: &[f32] = bytemuck::cast_slice(&buffer);
        buckets.push(bucket.to_vec());
    }
    println!("Loaded buckets: {buckets:?}");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::LoadPytorchModel => {
            load_pytorch_model()?;
        }
        Commands::UseF16 => use_f16()?,
        Commands::DumpModel => dump_model()?,
        Commands::DumpBuckets2Disk => dump_vector2disk()?,
        Commands::LoadBucketsFromDisk => load_vector_from_disk()?,
    }

    Ok(())
}
