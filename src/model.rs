use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        gru::{self, Gru, GruConfig},
        loss::{MseLoss, Reduction::Mean},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
    train::RegressionOutput,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
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
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            gru1: GruConfig::new(2, 32, true).init(device),
            gru2: GruConfig::new(32, 64, true).init(device),
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
        let x = self.gru1.forward(x, None);
        let x = self.dropout.forward(x);
        let x = self.gru2.forward(x, None);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        // Use dimensions from the LSTM output for slicing.
        let dims = x.dims();
        let x = x
            .slice([
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
