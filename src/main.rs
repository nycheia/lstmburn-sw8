mod model;
mod data;
mod training;
mod inference;
use std::{error::Error, path::Path};
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu}, module::Module, optim::AdamConfig, record::Recorder
};
use burn::record::{BinFileRecorder, FullPrecisionSettings};


fn main() -> Result<(), Box<dyn Error>> {

    let artifact_dir = "./training-logs";

    //let unformated_data:String = String::from("1745405818897,(57.0121349, 9.9908265),(57.0121448,9.9907374),(57.0121381, 9.9907434),(55.0121448,9.9907374),(55.0121381, 9.9907434),(57.0121349, 9.9908265),(57.0121349, 9.9908265),(57.0121349, 9.9908265),(57.0121349, 9.9908265),(57.0121349, 9.9908265),");
    //let unformated_data:String = String::from("1372636858620000589,(-8.618643,41.141412),(-8.618499,41.141376),(-8.620326,41.14251),(-8.622153,41.143815),(-8.623953,41.144373),(-8.62668,41.144778),(-8.627373,41.144697),(-8.630226,41.14521),(-8.632746,41.14692),(-8.631738,41.148225),(-8.629938,41.150385),(-8.62911,41.151213),(-8.629128,41.15124),(-8.628786,41.152203),(-8.628687,41.152374),(-8.628759,41.152518),(-8.630838,41.15268),(-8.632323,41.153022),(-8.631144,41.154489),(-8.630829,41.154507),(-8.630829,41.154516),(-8.630829,41.154498),(-8.630838,41.154489)");
    //let unformated_data:String = String::from("1372637303620000596,(-8.639847,41.159826),(-8.640351,41.159871),(-8.642196,41.160114),(-8.644455,41.160492),(-8.646921,41.160951),(-8.649999,41.161491),(-8.653167,41.162031),(-8.656434,41.16258),(-8.660178,41.163192),(-8.663112,41.163687),(-8.666235,41.1642),(-8.669169,41.164704),(-8.670852,41.165136),(-8.670942,41.166576),(-8.66961,41.167962),(-8.668098,41.168988),(-8.66664,41.170005),(-8.665767,41.170635),(-8.66574,41.170671)");

    let unformated_data:String = String::from("1372636858620000589,(0.0, 0.0),(1.0,1.0),(2.0,2.0),(3.0,3.0),(4.0,4.0),(5.0,5.0),(6.0,6.0),(7.0,7.0),(8.0,8.0),(9.0,9.0),");

    // Model config
    let hidden_size:usize = 256;
    // dropout should remain close to 0.5
    let dropout:f64 = 0.5;

    // Training config
    let num_epochs: usize = 2;
    let batch_size: usize = 64;
    let num_workers: usize = 1;
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

    let model_path = format!("{}/model.bin", artifact_dir);
    let model_path = Path::new(&model_path);

    /*
    // Create new model if not existing
    if !model_path.exists(){
        println!("Creating new model");
        model_config.init::<MyBackend>(&device);
    } 
     */  
    model_config.init::<MyBackend>(&device);

    let training_config = TrainingConfig{
        model: model_config,
        optimizer: optimizer_config,
        num_epochs,
        batch_size,
        num_workers,
        seed,
        learning_rate,
    };
    /*
    // Load existing model
    if model_path.exists(){
        println!("Loading existing model");
        let record = BinFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).expect("Trained model should exist");
        training_config.model.init::<MyBackend>(&device).load_record(record);
    }
     */

    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        training_config,
        device.clone(),
        unformated_data.clone(),
    );
    
    let (route, label ) = crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        unformated_data,
    );

    println!("Predicted route: {:?}", route);
    println!("Expected {:?}", label);
     

    Ok(())
}