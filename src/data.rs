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
                // Assume that the first N-1 columns are features and the last column is the label.
                let features = record[..record.len()-1]
                    .iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                let label = record
                    .last()
                    .unwrap_or(&"0".to_string())
                    .parse::<i64>()
                    .unwrap_or(0);
                CsvItem { features, label }
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
    pub labels: Tensor<B, 1, Int>,
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

impl<B: Backend> batcher::Batcher<CsvItem, CsvBatch<B>> for CsvBatcher<B>{
    fn batch(&self, items: Vec<CsvItem>) -> CsvBatch<B> {
        let features = items.iter().map(|item| {
            // Create a 2D tensor with shape [1, sequence_length]
            let data = TensorData::new(item.features.clone(), [1, item.features.len()]);
            Tensor::<B, 2>::from_data(data, &self.device)
        }).collect::<Vec<_>>();

        let labels = items.iter().map(|item| {
            Tensor::<B, 1, Int>::from_data([item.label.elem::<B::IntElem>()], &self.device)
        }).collect::<Vec<_>>();

        let features = Tensor::cat(features, 0).to_device(&self.device); // shape: [batch_size, sequence_length]
        let dims = features.dims(); // dims[0] is batch_size, dims[1] is 2.
        let features = features.reshape([dims[0], 1, dims[1]]);

        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        CsvBatch { features, labels }

    }
}


//burn book test code
/*#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1,28, 28])) //  no channel dimension.
            // Normalize: make between [0,1] and adjust mean=0, std=1.
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(item.label as i64).elem::<B::IntElem>()],
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MnistBatch { images, targets }
    }
}*/
