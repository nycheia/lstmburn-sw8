use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{
    TrainStep, TrainOutput, ValidStep, RegressionOutput, LearnerBuilder,
    metric::LossMetric,
};
use burn::config::Config;
use burn::optim::AdamConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::record::CompactRecorder;
use crate::model::Model;
use crate::ModelConfig;
use crate::data::{CsvDataset, CsvBatch, CsvBatcher};
use burn::module::Module;
use std::sync::Arc;

impl<B: AutodiffBackend> TrainStep<CsvBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: CsvBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.features, batch.labels);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CsvBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: CsvBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.features, batch.labels)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);
    
    let batcher_train = CsvBatcher::<B>::new(device.clone());
    let batcher_valid = CsvBatcher::<B::InnerBackend>::new(device.clone());
    let csv_dataset = CsvDataset::from_csv("./train.csv").expect("Failed to load CSV data");
    let (train_dataset, test_dataset) = csv_dataset.split(0.7);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(train_dataset));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(test_dataset));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
