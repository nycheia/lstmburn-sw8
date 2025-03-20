use burn::{
    nn::{
        lstm::{Lstm, LstmConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    lstm1: Lstm<B>,
    lstm2: Lstm<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            lstm1: LstmConfig::new(2, 32, true).init(device),
            lstm2: LstmConfig::new(32, 64, true).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(64, 32).init(device),
            linear2: LinearConfig::new(32, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_batch_size, _sequence_length, _feature_size] = features.dims();

        let x = features.clone();
        let (x, _) = self.lstm1.forward(x, None);
        let x = self.dropout.forward(x);
        let (x, _) = self.lstm2.forward(x, None);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        // Use dimensions from the LSTM output for slicing.
        let dims = x.dims();
        let x = x.slice([
            0..dims[0],             // batch dimension
            (dims[1] - 1)..dims[1], // last time step
            0..dims[2],             // all features
        ])
        .squeeze(1); // Now x should have shape [batch_size, hidden_dim]

        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}