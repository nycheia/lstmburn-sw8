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
            lstm1: LstmConfig::new(1, 32, true).init(device),
            lstm2: LstmConfig::new(2, 64, true).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, sequence_length, feature_size] = images.dims();

        // Create a channel at the second dimension.
        let x = images;


        let (x, _) = self.lstm1.forward(x, None);
        let x = self.dropout.forward(x);
        let (x,_) = self.lstm2.forward(x, None); 
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        //let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        //let x = x.reshape([batch_size, 16 * 8 * 8]);
        let last_index: Tensor<B, 1, Int> = Tensor::from([sequence_length as i32 - 1]);
        let x = x.select(1, last_index);

        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = x.squeeze(1);
        self.linear2.forward(x) // [batch_size, num_classes]
    }
}