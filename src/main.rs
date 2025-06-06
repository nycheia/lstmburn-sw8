mod model;
mod data;
mod training;
mod inference;
//use std::collections::HashSet;
use std::error::Error;
use std::fs;
use burn::data::dataloader::Dataset;
//use csv::ReaderBuilder;
//use serde_json::Value;
use crate::{model::ModelConfig, training::TrainingConfig};
//use data::dataset::InMemDataset;
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use crate::data::CsvDataset;

// Trajectory structure.
/*#[derive(Debug)]
struct Trajectory {
    id: String,
    timestamp: String,
    points: Vec<Vec<f64>>,
}

// Parse a polyline string along with id and timestamp into a Trajectory.
fn parse_trajectory_from_polyline(polyline: &str, id: &str, timestamp: &str) -> Result<Trajectory, Box<dyn Error>> {
    // The polyline is expected to be a JSON string like "[[-8.618643,41.141412], ...]".
    let points: Vec<Vec<f64>> = serde_json::from_str(polyline)?;
    Ok(Trajectory {
        id: id.to_string(),
        timestamp: timestamp.to_string(),
        points,
    })
}

// If your JSON contains multiple trajectories, parse them.
fn parse_trajectories(data: &str) -> Result<Vec<Trajectory>, Box<dyn Error>> {
    let parsed: Value = serde_json::from_str(data)?;
    let arr = parsed.as_array().ok_or("Expected an array")?;
    
    let mut trajectories = Vec::new();
    for traj in arr {
        let id = traj.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = traj.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        // Here we assume the "points" field is a JSON array; convert it to a string.
        let polyline = traj.get("points").ok_or("Missing 'points' field")?.to_string();
        let trajectory = parse_trajectory_from_polyline(&polyline, id, timestamp)?;
        trajectories.push(trajectory);
    }
    Ok(trajectories)
}

// CSV dataset loader that filters out rows with "MISSING_DATA" = true
// and retains only selected columns.
struct InMemDataset {
    records: Vec<Vec<String>>,
    skipped: Vec<Vec<String>>,
}

impl InMemDataset {
    fn from_csv(path: &str, reader_builder: ReaderBuilder) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut rdr = reader_builder.from_reader(file);
        let headers = rdr.headers()?;
        println!("Column names: {:?}", headers);

        // Define columns to remove.
        let remove_cols: HashSet<&str> = [
            "CALL_TYPE",
            "ORIGIN_CALL",
            "ORIGIN_STAND",
            "TAXI_ID",
            "DAY_TYPE",
            "MISSING_DATA"
        ].iter().cloned().collect();

        // Keep columns not in remove_cols.
        let keep_cols: Vec<usize> = headers.iter().enumerate()
            .filter(|(_i, col)| !remove_cols.contains(*col))
            .map(|(i, _)| i)
            .collect();

        let kept_columns: Vec<_> = keep_cols.iter()
            .map(|&i| headers.get(i).unwrap_or(""))
            .collect();
        println!("Keeping columns: {:?}", kept_columns);

        // Locate the "MISSING_DATA" column if it exists.
        let missing_data_idx = headers.iter().position(|col| col == "MISSING_DATA");

        let mut records: Vec<Vec<String>> = Vec::new();
        let mut skipped_records: Vec<Vec<String>> = Vec::new();
        let mut missing_count = 0;

        for result in rdr.records() {
            let record = result?;
            if let Some(idx) = missing_data_idx {
                if record.get(idx).unwrap_or("").trim().eq_ignore_ascii_case("true") {
                    missing_count += 1;
                    let skipped: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                    skipped_records.push(skipped);
                    continue;
                }
            }
            let new_record: Vec<String> = keep_cols.iter()
                .map(|&i| record.get(i).unwrap_or("").to_string())
                .collect();
            records.push(new_record);
        }

        println!("Removed {} records with MISSING_DATA = true.", missing_count);
        println!("After filtering, {} records remain.", records.len());

        Ok(Self { records, skipped: skipped_records })
    }
}*/

fn main() -> Result<(), Box<dyn Error>> {
    // Load CSV data.
    /*let mut builder = ReaderBuilder::new();
    builder.delimiter(b',');
    let file_path = "./train.csv";
    // Now, builder still owns a ReaderBuilder that can be passed by value.
    let dataset = InMemDataset::from_csv(
        file_path,
        builder,
    )?;
    
    println!("Loaded {} records", dataset.records.len());*/

    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(128).init::<MyBackend>(&device);

    println!("{}", model);

    type MyAutodiffBackend = Autodiff<MyBackend>;
    let artifact_dir = "./training-logs";
    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(128), AdamConfig::new()),
        device.clone(),
    );

    /*let csv_dataset = CsvDataset::from_csv("./train.csv").expect("Failed to load CSV data");
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        csv_dataset
            .get(1)
            .unwrap(),
    );*/
    let mut datasets = Vec::new();
    for entry in fs::read_dir("aalborgopenstreetmapdata")? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("csv") {
            let ds = CsvDataset::from_csv(path.to_str().unwrap())
                .expect(&format!("Failed to load {:?}", path));
            datasets.push(ds);
        }
    }

    for (i, ds) in datasets.into_iter().enumerate() {
        let record = ds.get(1)
            .unwrap_or_else(|| panic!("Dataset {} has no record #1", i));
        crate::inference::infer::<MyBackend>(
            artifact_dir,
            device.clone(),
            record,
        );
        println!("File #{:?},", i);
    }

    /*for (i, item) in csv_dataset.items.iter().enumerate() {
        if i >= 5 { break; }
        println!("Item {}: {:?}", i, item);
    }*/

    Ok(())
}
