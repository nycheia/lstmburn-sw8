mod model;
mod data;
mod training;
mod inference;
use std::error::Error;
use burn::data::dataloader::Dataset;
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use crate::data::CsvDataset;


fn main() -> Result<(), Box<dyn Error>> {

    // Model config
    let hidden_size:usize = 128;
    let dropout:f64 = 0.5;

    // Training config
    let num_epochs: usize   = 1;
    let batch_size: usize = 4;
    let num_workers: usize = 42;
    let seed: u64 = 42;
    let learning_rate: f64 = 1.0e-4;
    let optimizer_config = AdamConfig::new();


    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();

    let model_config = ModelConfig{
        hidden_size,
        dropout,
    };

    let model = model_config.init::<MyBackend>(&device);

    println!("{}", model);

    type MyAutodiffBackend = Autodiff<MyBackend>;
    let artifact_dir = "./training-logs";
    let device = burn::backend::wgpu::WgpuDevice::default();


    let training_config = TrainingConfig{
        model: model_config,
        optimizer: optimizer_config,
        num_epochs,
        batch_size,
        num_workers,
        seed,
        learning_rate,
    };


    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        training_config,
        device.clone(),
    );

    

    let csv_dataset = CsvDataset::from_csv("./train.csv").expect("Failed to load CSV data");
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        csv_dataset
            .get(1)
            .unwrap(),
    );

    /*     for (i, item) in csv_dataset.items.iter().enumerate() {
        if i >= 5 { break; }
        println!("Item {}: {:?}", i, item);
    }
    */

    Ok(())
}