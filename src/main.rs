mod model;
mod data;
mod training;
mod inference;
use std::error::Error;
//use std::fs::{File, read_to_string};
use burn::data::dataloader::Dataset;
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use crate::data::CsvDataset;


fn main() -> Result<(), Box<dyn Error>> {
    
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{}", model);

    type MyAutodiffBackend = Autodiff<MyBackend>;
    let artifact_dir = "./training-logs";
    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
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

    for (i, item) in csv_dataset.items.iter().enumerate() {
        if i >= 5 { break; }
        println!("Item {}: {:?}", i, item);
    }

    Ok(())
}