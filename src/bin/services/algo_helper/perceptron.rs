use crate::environments::env::Forward;
use burn::nn;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct MyQmlp<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    linear4: nn::Linear<B>,
    output:  nn::Linear<B>,
}

impl<B: Backend> MyQmlp<B> {
    pub(crate) fn new(device: &B::Device, in_features: usize, out_actions: usize) -> Self {
        let linear1 = nn::LinearConfig::new(in_features, 1024).with_bias(true).init(device);
        let linear2 = nn::LinearConfig::new(1024, 512).with_bias(true).init(device);
        let linear3 = nn::LinearConfig::new(512, 512).with_bias(true).init(device);
        let linear4 = nn::LinearConfig::new(512, 128).with_bias(true).init(device);
        let output  = nn::LinearConfig::new(128, out_actions).with_bias(true).init(device);
        Self { linear1, linear2, linear3, linear4, output }
    }
}

impl<B: Backend> Forward for MyQmlp<B> {
    type B = B;

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(input).tanh();
        let x = self.linear2.forward(x).tanh();
        let x = self.linear3.forward(x).tanh();
        let x = self.linear4.forward(x).tanh();
        self.output.forward(x)
    }
}

pub fn get_device() -> burn::backend::libtorch::LibTorchDevice {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = args.iter().any(|arg| arg == "--gpu");

    if tch::Cuda::is_available() && use_gpu {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    } else {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }
}