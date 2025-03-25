use crate::TrainingConfig;
use crate::data::{CsvItem, CsvBatcher};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::data::dataloader::batcher::Batcher;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: CsvItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);
    let fixed_len = 10;
    let mut route = item.label.clone();
    if route.len() < fixed_len {
        let pad_value = route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);
        route.resize(fixed_len, pad_value);
    } else if route.len() > fixed_len {
        route.truncate(fixed_len);
    }

    let label = route;
    let batcher = CsvBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.features);
    //let predicted_route = output.squeeze::<2>(1).into_data();
    use std::convert::TryInto;

    let tensor_data = output.into_data();
    let raw_bytes = tensor_data.as_bytes(); // assuming this gives you a Vec<u8>

    let f32_values: Vec<f32> = raw_bytes
        .chunks(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect();

    // Now, to get the route as a vector of coordinate pairs:
    let route: Vec<Vec<f32>> = f32_values.chunks(2).map(|chunk| chunk.to_vec()).collect();
    println!("Predicted route: {:?}", route);
    println!("Expected {:?}", label);
}