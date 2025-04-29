use burn::prelude::*;
use burn::data::dataloader::batcher;
//use burn::data::dataset::InMemDataset;
use burn::tensor::TensorData;
//use burn::prelude::{Backend, Tensor, Int};

use burn::data::dataset::Dataset;

/// A custom item type representing one row from your CSV.
#[derive(Debug, Clone)]
pub struct Item {
    pub features: Vec<f64>,        // e.g., [timestamp, ...]
    pub label: Vec<Vec<f64>>,      // the coordinates, e.g., [[lon, lat], ...]
}


impl Dataset<Item> for ItemDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<Item> {
        self.items.get(index).cloned()
    }
}

/// A custom dataset that converts CSV records into Item instances.
#[derive(Debug)]
pub struct ItemDataset {
    pub items: Vec<Item>,
}

impl ItemDataset {
    pub fn create_dataset(data: Vec<Vec<f64>>, train_ratio: f64) -> (ItemDataset, ItemDataset) {
        if data.len() < 2 {
            return (ItemDataset { items: vec![] }, ItemDataset { items: vec![] });
        }

        // Get the timestamp from the first element (index 0)
        let timestamp = data[0][0];
        let coordinates = &data[1..]; // coordinates are everything after the timestamp

        let mut items = vec![];

        // Loop through the coordinates and create items
        for i in 0..(coordinates.len()) {
            let _input = coordinates[i..i].to_vec(); // seq_len coordinates
            let target = coordinates[i].clone();    // the next coordinate (target)

            let item = Item {
                features: vec![timestamp], // timestamp for each sequence
                label: vec![target],       // label: the next coordinate in the sequence
            };
            items.push(item);
        }

        // Ensure train_ratio is between 0 and 1
        let train_ratio = train_ratio.clamp(0.0, 1.0);

        // Calculate the split index based on the ratio
        let split_idx = (items.len() as f64 * train_ratio).floor() as usize;

        // Split the data sequentially into train and test datasets
        let (train, test) = items.split_at(split_idx);

        (
            ItemDataset { items: train.to_vec() },  // Train dataset
            ItemDataset { items: test.to_vec() },   // Test dataset
        )
    }
}

/// A batch produced from CSV data. The features are arranged as a 3D tensor
/// with shape [batch_size, sequence_length, feature_size] and labels as a 1D tensor.
#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub features: Tensor<B, 3>,
    pub labels: Tensor<B, 3>,
}

/// A batcher that converts Item instances into Batch instances.
#[derive(Clone)]
pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> batcher::Batcher<Item, Batch<B>> for DataBatcher<B> {
    fn batch(&self, items: Vec<Item>) -> Batch<B> {
        let fixed_len = 10;

        // Process features (timestamp + route coordinates)
        let features = items.iter().map(|item| {
            let mut seq = vec![];

            // Extract the timestamp (assuming it's the first feature)
            let timestamp = item.features[0];

            // Process the route coordinates (lat, lon)
            let route = item.label.clone();
            let mut fixed_route = route.clone();

            // Pad or truncate the route to match the fixed length
            if fixed_route.len() < fixed_len {
                let pad_value = fixed_route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);
                fixed_route.resize(fixed_len, pad_value);
            } else if fixed_route.len() > fixed_len {
                fixed_route.truncate(fixed_len);
            }

            // Combine timestamp with each coordinate (lat, lon)
            for coord in fixed_route {
                seq.push(vec![timestamp, coord[0], coord[1]]);
            }

            // Flatten the sequence to create the tensor (flattening timestamp, lat, lon into a flat vector)
            let data = TensorData::new(seq.into_iter().flatten().collect::<Vec<f64>>(), [fixed_len, 3]);

            // Reshape to [1, fixed_len, 3] where the sequence has 3 features at each timestep
            Tensor::<B, 2>::from_data(data, &self.device).reshape([1, fixed_len, 3])
        }).collect::<Vec<_>>();

        // Concatenate all feature tensors along the batch dimension (axis 0)
        let features = Tensor::cat(features, 0).to_device(&self.device);

        // Process labels similarly (lat, lon coordinates)
        let labels = items.iter().map(|item| {
            let route = item.label.clone();
            let mut fixed_route = route.clone();
            
            // Pad or truncate the route to match the fixed length
            if fixed_route.len() < fixed_len {
                let pad_value = fixed_route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);
                fixed_route.resize(fixed_len, pad_value);
            } else if fixed_route.len() > fixed_len {
                fixed_route.truncate(fixed_len);
            }

            // Flatten the coordinates (lat, lon) into a vector for each route
            let flat_label: Vec<f64> = fixed_route.into_iter().flatten().collect();

            // Create a tensor from the label sequence
            let data = TensorData::new(flat_label, [fixed_len, 2]);

            // Reshape to [1, fixed_len, 2] for lat/lon coordinates
            Tensor::<B, 2>::from_data(data, &self.device).reshape([1, fixed_len, 2])
        }).collect::<Vec<_>>();

        // Concatenate all label tensors along the batch dimension
        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        // Return the batch with features and labels
        Batch { features, labels }
    }
}


pub fn format_string(data: &str) -> Vec<Vec<f64>> {
    let data = data.trim_end_matches(',');

    let mut parts = data.splitn(2, ','); // separate timestamp and rest
    let timestamp = parts
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|n| vec![n])
        .unwrap_or_else(|| vec![]);

    let coordinates = parts
        .next()
        .unwrap_or("")
        .split("),(")
        .map(|s| {
            let clean = s.trim_matches(|c| c == '(' || c == ')');
            clean
                .split(',')
                .map(|n| n.trim().parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        });

    let mut result: Vec<Vec<f64>> = vec![timestamp];
    result.extend(coordinates);
    result
}

