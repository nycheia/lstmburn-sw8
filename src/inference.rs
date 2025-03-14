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

    let label = item.label;
    let batcher = CsvBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.features);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}