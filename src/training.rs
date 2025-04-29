use crate::ModelConfig;
use crate::data::{format_string, Batch, DataBatcher, ItemDataset};
use crate::model::Model;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{
    LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
};
use std::sync::Arc;

impl<B: AutodiffBackend> TrainStep<Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.features, batch.labels);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Batch<B>) -> RegressionOutput<B> {
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

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    unformated_data: String,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = DataBatcher::<B>::new(device.clone());
    let batcher_valid = DataBatcher::<B::InnerBackend>::new(device.clone());
    let formated_data = format_string(&unformated_data);
    println!("Formated String: {:?}", formated_data);
    println!("");
    let (train_dataset, test_dataset) = ItemDataset::create_dataset(formated_data, 0.7);

    println!("train: {:?}", train_dataset);
    println!("");
    println!("test: {:?}", test_dataset);


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
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
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
        .save_file(
            format!("{artifact_dir}/model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Trained model should be saved successfully");
}

