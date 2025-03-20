use burn::{
    nn::{
        lstm::{Lstm, LstmConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
    train::RegressionOutput,
};
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction::Mean;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    output_dim: usize,
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
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let output_dim = 2;
        Model {
            output_dim,
            lstm1: LstmConfig::new(1, 32, true).init(device),
            lstm2: LstmConfig::new(32, 64, true).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(64,  32).init(device),
            linear2: LinearConfig::new(32, output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = features.clone();
        let (x, _) = self.lstm1.forward(x, None);
        let x = self.dropout.forward(x);
        let (x, _) = self.lstm2.forward(x, None);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        // x now has shape [batch_size, seq_len, hidden_dim] (e.g., [64, 10, 64])
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let hidden_dim = dims[2];
        
        // Flatten batch and time dimensions to apply linear layers to each time step:
        let reshaped = x.reshape([batch_size * seq_len, hidden_dim]); // [batch_size * seq_len, hidden_dim]
        let out = self.linear1.forward(reshaped);
        let out = self.dropout.forward(out);
        let out = self.activation.forward(out);
        let out = self.linear2.forward(out); // expected shape: [batch_size * seq_len, output_dim]
        
        out.reshape([batch_size, seq_len, self.output_dim])
    }
    
    pub fn forward_regression(
        &self,
        features: Tensor<B, 3>,
        targets: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
        let output_seq = self.forward(features); // shape: [batch_size, seq_len, output_dim]
        let dims = output_seq.dims(); // dims = [batch_size, seq_len, output_dim]
        let batch_size = dims[0];
        let seq_len = dims[1];
        let output_dim = dims[2];
        let target_shape = [batch_size * seq_len, output_dim]; // should be [640, 2]
        let output_2d = output_seq.reshape(target_shape);
        let targets_2d = targets.reshape(target_shape);
        let loss = MseLoss::new().forward(output_2d.clone(), targets_2d.clone(), Mean);
        RegressionOutput::new(loss, output_2d, targets_2d)
    }
}
