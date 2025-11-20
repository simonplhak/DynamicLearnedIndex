use std::{collections::HashMap, path::Path};

use anyhow::Result;
use candle_core::{DType, Device, Result as CandleResult, Tensor, safetensors};
use candle_nn::{Linear, Module, VarBuilder};
use clap::{Parser, Subcommand, command};

#[derive(Parser, Debug)]
#[command(name = "cli_app", version = "1.0", about = "CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    LoadPytorchModel,
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
        diff.flatten(0, 1)?.to_vec1()?.iter().sum::<f32>()
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::LoadPytorchModel => {
            load_pytorch_model()?;
        }
    }

    Ok(())
}
