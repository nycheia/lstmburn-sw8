use crate::TrainingConfig;
use crate::data::{format_string, DataBatcher, Item};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::record::Recorder;
use burn::data::dataloader::batcher::Batcher;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, unformated_string: String) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    
    let formated_string = format_string(&unformated_string);

    // Prepare the data for prediction
    let item = Item {
        features: formated_string.first().cloned().unwrap_or_default(), // Take the first vec as features
        label: formated_string.iter().skip(1).cloned().collect(),       // The rest as the label
    };

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    // Ensure the label size is 1 (i.e., only one future point to predict)
    let fixed_len = 10; // Assuming this is the length of the sequence in the input data
    let mut route = item.label.clone();
    if route.len() < fixed_len {
        let pad_value = route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);
        route.resize(fixed_len, pad_value);
    } else if route.len() > fixed_len {
        route.truncate(fixed_len);
    }

    // The last entry in the route will be the actual target value for comparison
    let actual_next_point = route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);

    // Remove the last entry from the input sequence for prediction
    let input_data = formated_string[..formated_string.len() - 1].to_vec(); // Remove the last point

    // Prepare the item with the modified input
    let item = Item {
        features: input_data.first().cloned().unwrap_or_default(),
        label: input_data.iter().skip(1).cloned().collect(),
    };

    // Batch the data (this assumes batch size of 1)
    let batcher = DataBatcher::new(device);
    let batch = batcher.batch(vec![item]);

    // Perform the forward pass to predict the next point
    let output = model.forward(batch.features);

    // Convert output tensor data into a Vec<f64>
    use std::convert::TryInto;
    let tensor_data = output.into_data();
    let raw_bytes = tensor_data.as_bytes(); // Assuming this gives you a Vec<u8>

    let f64_values: Vec<f64> = raw_bytes
        .chunks(8)
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            f64::from_le_bytes(arr)
        })
        .collect();

    // The output now contains only one predicted point
    let predicted_point = f64_values.chunks(2).next().unwrap_or(&[0.0, 0.0]);

    // Return the predicted point and the actual next point (label)
    let predicted_route = vec![predicted_point.to_vec()];

    (predicted_route, vec![actual_next_point])
}
