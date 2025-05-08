use burn::nn;
use burn::prelude::*;
use crate::config::hidden_sizes;

pub trait Forward {
    type B: Backend;
    fn forward<const DIM: usize>(&self, input: Tensor<Self::B, DIM>) -> Tensor<Self::B, DIM>;
}

#[derive(Module, Debug)]
pub struct MyQmlp<B: Backend> {
    hidden_layers: Vec<nn::Linear<B>>,
    output: nn::Linear<B>,
}

impl<B: Backend> MyQmlp<B> {
    pub(crate) fn new(device: &B::Device, in_features: usize, out_actions: usize) -> Self {
        let mut layers = Vec::new();
        let mut input_size = in_features;

        for hidden_size in hidden_sizes() {
            let layer = nn::LinearConfig::new(input_size, hidden_size).with_bias(true).init(device);
            layers.push(layer);
            input_size = hidden_size;
        }

        let output = nn::LinearConfig::new(input_size, out_actions).with_bias(true).init(device);

        Self {
            hidden_layers: layers,
            output,
        }
        /*let linear1 = nn::LinearConfig::new(in_features, 2048).with_bias(true).init(device);
        let linear2 = nn::LinearConfig::new(2048, 1024).with_bias(true).init(device);
        let linear3 = nn::LinearConfig::new(1024, 512).with_bias(true).init(device);
        let linear4 = nn::LinearConfig::new(512, 256).with_bias(true).init(device);
        let output  = nn::LinearConfig::new(256, out_actions).with_bias(true).init(device);
        Self { linear1, linear2, linear3, linear4, output }*/
    }
}

impl<B: Backend> Forward for MyQmlp<B> {
    type B = B;

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = input;
        for layer in &self.hidden_layers {
            x = layer.forward(x).tanh();
        }
        self.output.forward(x)
    }
}
