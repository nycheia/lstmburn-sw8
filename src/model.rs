use burn::{
    nn::{
        gru::{Gru, GruConfig},
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
    gru1: Gru<B>,
    gru2: Gru<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub hidden_size: usize,
    #[config(default = "0.5")]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let output_dim = 2;

        Model {
            output_dim,
            gru1: GruConfig::new(1, 32, true).init(device),
            gru2: GruConfig::new(32, 64, true).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(64, 32).init(device),
            linear2: LinearConfig::new(32, output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = features.clone();
    
        // GRU 1: Process the input through the first GRU layer
        let gru1_output = self.gru1.forward(x, None); 
    
        // Apply dropout on the output of GRU 1
        let x = self.dropout.forward(gru1_output); 
    
        // GRU 2: Process the output of GRU 1 through the second GRU layer
        let gru2_output = self.gru2.forward(x, None); 
    
        // Apply dropout on the output of GRU 2
        let x = self.dropout.forward(gru2_output);
    
        // Apply activation after dropout
        let x = self.activation.forward(x);
    
        // Now `x` has shape [batch_size, seq_len, hidden_dim]
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let hidden_dim = dims[2];
    
        // Flatten the batch and sequence dimensions for the linear layers
        let reshaped = x.reshape([batch_size * seq_len, hidden_dim]); // [batch_size * seq_len, hidden_dim]
    
        // Pass through the linear layers
        let out = self.linear1.forward(reshaped);
        let out = self.dropout.forward(out);
        let out = self.activation.forward(out);
        let out = self.linear2.forward(out); // Expected shape: [batch_size * seq_len, output_dim]
    
        // Reshape the output to [batch_size, seq_len, output_dim]
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
