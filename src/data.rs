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
        if data.is_empty() {
            return (ItemDataset { items: vec![] }, ItemDataset { items: vec![] });
        }

        // Extract the feature (first element)
        let features = data[0]
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();

        // Extract the label (remaining elements)
        let label = data[1..]
            .iter()
            .map(|pair| {
                pair.iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        // Calculate the split index
        let train_len = (label.len() as f64 * train_ratio).round() as usize;

        // Split the label into training and testing
        let train_labels = label[..train_len].to_vec();
        let test_labels = label[train_len..].to_vec();

        // Create the training and testing datasets
        let train_item = Item { features: features.clone(), label: train_labels };
        let test_item = Item { features: features.clone(), label: test_labels };

        // Return the datasets with their items
        (
            ItemDataset { items: vec![train_item] },
            ItemDataset { items: vec![test_item] }
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
        // Define a fixed route length
        let fixed_len = 10;
        
        // Process features as before
        let features = items.iter().map(|item| {
            // Suppose item.features currently is a vector with a single timestamp.
            // To create a sequence of length 10, you might replicate that timestamp 10 times:
            let mut seq = vec![];
            for _ in 0..10 {
                seq.push(item.features[0]);
            }
            // Now, seq has 10 elements. Create a tensor with shape [10, 1]:
            let data = TensorData::new(seq, [10, 1]);
            Tensor::<B, 2>::from_data(data, &self.device)
        }).collect::<Vec<_>>();

        // Process labels: pad or truncate each route to fixed_len coordinate pairs
        let labels = items.iter().map(|item| {
            let mut route = item.label.clone();
            let current_len = route.len();
            if current_len < fixed_len {
                // If route is too short, pad it by repeating the last coordinate pair
                let pad_value = route.last().cloned().unwrap_or_else(|| vec![0.0, 0.0]);
                route.resize(fixed_len, pad_value);
            } else if current_len > fixed_len {
                // If route is too long, truncate it
                route.truncate(fixed_len);
            }
            // Now route has exactly fixed_len coordinate pairs
            let rows = route.len();  // This should be fixed_len
            let cols = if rows > 0 { route[0].len() } else { 0 }; // Expected to be 2
            let flat_label: Vec<f64> = route.into_iter().flatten().collect();
            let data = TensorData::new(flat_label, [rows, cols]);
            // Convert to a tensor of shape [1, fixed_len, 2]
            Tensor::<B, 2>::from_data(data, &self.device).reshape([1, rows, cols])
        }).collect::<Vec<_>>();

        let features = Tensor::cat(features, 0).to_device(&self.device); // shape: [batch_size, feature_length]
        let dims = features.dims();
        let features = features.reshape([dims[0], 1, dims[1]]); // [batch_size, 1, feature_length]
        let labels = Tensor::cat(labels, 0).to_device(&self.device); // shape: [batch_size, fixed_len, 2]

        //println!("Batch features shape: {:?}", features.dims());
        //println!("Batch labels shape: {:?}", labels.dims());
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

