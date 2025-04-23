mod model;
mod data;
mod training;
mod inference;
use std::error::Error;
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu}, module::Module, optim::AdamConfig, record::Recorder
};
use burn::record::{BinFileRecorder, FullPrecisionSettings};


fn main() -> Result<(), Box<dyn Error>> {

    let artifact_dir = "./training-logs";

    let unformated_data:String = String::from("1745405818897,(57.0121349, 9.9908265),(57.0121448,9.9907374),(57.0121381, 9.9907434),(55.0121448,9.9907374),(55.0121381, 9.9907434),");


    // Model config
    let hidden_size:usize = 256;
    // dropout should remain close to 0.5
    let dropout:f64 = 0.5;

    // Training config
    let num_epochs: usize = 2;
    let batch_size: usize = 64;
    let num_workers: usize = 4;
    // Seed ensures reproducibility
    let seed: u64 = 42;
    let learning_rate: f64 = 1.0e-4;
    // Optimizer should not by changed, but can be if needed
    let optimizer_config = AdamConfig::new();

    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();

    
    let model_config = ModelConfig{
        hidden_size,
        dropout,
    };

    // Should be used to create a initial model
    //model_config.init::<MyBackend>(&device);

    let training_config = TrainingConfig{
        model: model_config,
        optimizer: optimizer_config,
        num_epochs,
        batch_size,
        num_workers,
        seed,
        learning_rate,
    };

    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    
    // Load existing model
    training_config.model.init::<MyBackend>(&device).load_record(record);

    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        training_config,
        device.clone(),
        unformated_data,
    );

    

    Ok(())
}