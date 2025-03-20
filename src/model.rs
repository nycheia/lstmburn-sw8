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
    // For regression, output_dim is the number of continuous values per time step.
    // For example, if you want to predict a coordinate pair, set output_dim to 2.
    #[config(default = "2")]
    output_dim: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    /// Note: Since Model does not store its config, we assume here that output_dim is fixed (e.g. 2).
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // Assuming each CSV record provides 1 input feature (timestamp)
            lstm1: LstmConfig::new(1, 32, true).init(device),
            lstm2: LstmConfig::new(32, 64, true).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(64, 32).init(device),
            // We set the final output dimension to 2 (a coordinate pair)
            linear2: LinearConfig::new(32, self.output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// Forward pass for sequence prediction.
    /// Expects `features` with shape [batch_size, seq_len, feature_size]
    /// and returns output with shape [batch_size, seq_len, output_dim].
    /// (We assume here that all sequences are padded to the same length.)
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        // Process the input through the LSTM layers.
        let x = features.clone();
        let (x, _) = self.lstm1.forward(x, None);
        let x = self.dropout.forward(x);
        let (x, _) = self.lstm2.forward(x, None);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        // x now has shape [batch_size, seq_len, hidden_dim] (e.g. [64, seq_len, 64])
        let dims = x.dims();
        // Flatten batch and time dimensions to apply linear layers to each time step:
        let reshaped = x.reshape([dims[0] * dims[1], dims[2]]);
        let out = self.linear1.forward(reshaped);
        let out = self.dropout.forward(out);
        let out = self.activation.forward(out);
        let out = self.linear2.forward(out); // shape: [batch_size * seq_len, output_dim]
        // Now, explicitly retrieve batch size and sequence length:
        let batch_size = dims[0];
        let seq_len = dims[1];
        // Reshape the linear layer output back to [batch_size, seq_len, output_dim]:
        out.reshape([batch_size, seq_len, 2])
    }
            
    pub fn forward_regression(
        &self,
        features: Tensor<B, 3>,
        targets: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
        let output_seq = self.forward(features); // shape: [batch_size, seq_len, output_dim]
        let dims = output_seq.dims(); // dims = [batch_size, seq_len, output_dim]
        // Flatten the batch and sequence dimensions.
        let output_2d = output_seq.reshape([dims[0] * dims[1], dims[2]]);
        let targets_2d = targets.reshape([dims[0] * dims[1], dims[2]]);
        let loss = MseLoss::new().forward(output_2d.clone(), targets_2d.clone(), Mean);
        RegressionOutput::new(loss, output_2d, targets_2d)
    }
    
}
