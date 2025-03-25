use burn::prelude::*;
use burn::data::dataloader::batcher;
//use burn::data::dataset::InMemDataset;
use burn::tensor::TensorData;
//use burn::prelude::{Backend, Tensor, Int};

use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use burn::data::dataset::Dataset;
use serde_json;

/// A custom item type representing one row from your CSV.
#[derive(Debug, Clone)]
pub struct CsvItem {
    pub features: Vec<f32>,        // e.g., [timestamp, ...]
    pub label: Vec<Vec<f32>>,      // the coordinates, e.g., [[lon, lat], ...]
}


impl Dataset<CsvItem> for CsvDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<CsvItem> {
        self.items.get(index).cloned()
    }
}

/// A simple in-memory dataset that holds CSV records.
pub struct InMemDataset {
    pub records: Vec<Vec<String>>,
}

impl InMemDataset {
    pub fn from_csv(path: &str, reader_builder: ReaderBuilder) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut rdr = reader_builder.from_reader(file);
        let headers = rdr.headers()?;
        
        // Define columns to remove.
        let remove_cols: HashSet<&str> = [
            "CALL_TYPE",
            "ORIGIN_CALL",
            "ORIGIN_STAND",
            "TAXI_ID",
            "DAY_TYPE",
            "MISSING_DATA",
        ]
        .iter()
        .cloned()
        .collect();
        
        // Keep columns not in remove_cols.
        let keep_cols: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter(|(_i, col)| !remove_cols.contains(*col))
            .map(|(i, _)| i)
            .collect();
        
        let mut records: Vec<Vec<String>> = Vec::new();
        // Locate the "MISSING_DATA" column if it exists.
        let missing_data_id = headers.iter().position(|col| col == "MISSING_DATA");
        
        for result in rdr.records() {
            let record = result?;
            // If MISSING_DATA is true, skip this record.
            if let Some(idx) = missing_data_id {
                if record.get(idx).unwrap_or("").trim().eq_ignore_ascii_case("true") {
                    continue;
                }
            }
            let new_record: Vec<String> = keep_cols
                .iter()
                .map(|&i| record.get(i).unwrap_or("").to_string())
                .collect();
            records.push(new_record);
        }
        Ok(Self { records })
    }
}

/// A custom dataset that converts CSV records into CsvItem instances.
pub struct CsvDataset {
    pub items: Vec<CsvItem>,
}

impl CsvDataset {
    /// Splits the dataset into two parts, the first containing `train_ratio` portion of the data.
    pub fn split(self, train_ratio: f64) -> (Self, Self) {
        let total = self.items.len();
        let train_len = (total as f64 * train_ratio).round() as usize;
        let (train_items, test_items) = self.items.split_at(train_len);
        (
            CsvDataset { items: train_items.to_vec() },
            CsvDataset { items: test_items.to_vec() },
        )
    }

    pub fn from_csv(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut builder = ReaderBuilder::new();
        builder.delimiter(b',');
        let dataset = InMemDataset::from_csv(path, builder)?;
        let items = dataset
            .records
            .into_iter()
            .map(|record| {
                // Suppose the first column is an ID (which we might ignore or transform),
                // the second column is the timestamp (which we'll use as our feature),
                // and the third column is the JSON string with coordinates.
                let timestamp_feature = record.get(1)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                // Parse the coordinates from the JSON string:
                let coords: Vec<Vec<f32>> = serde_json::from_str(record.get(2).unwrap_or(&"[]".to_string()))
                    .unwrap_or_default();
                CsvItem {
                    features: vec![timestamp_feature], // or include more features if available
                    label: coords,
                }
            })
            .collect();
        Ok(Self { items })
    }
}

/// A batch produced from CSV data. The features are arranged as a 3D tensor
/// with shape [batch_size, sequence_length, feature_size] and labels as a 1D tensor.
#[derive(Clone, Debug)]
pub struct CsvBatch<B: Backend> {
    pub features: Tensor<B, 3>,
    pub labels: Tensor<B, 3>,
}

/// A batcher that converts CsvItem instances into CsvBatch instances.
#[derive(Clone)]
pub struct CsvBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CsvBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> batcher::Batcher<CsvItem, CsvBatch<B>> for CsvBatcher<B> {
    fn batch(&self, items: Vec<CsvItem>) -> CsvBatch<B> {
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
            let flat_label: Vec<f32> = route.into_iter().flatten().collect();
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
        CsvBatch { features, labels }
    }
}


