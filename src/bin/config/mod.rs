use burn::backend::{Autodiff, LibTorch};
use burn::prelude::Backend;

pub type MyBackend = LibTorch;
pub type MyDevice = <MyBackend as Backend>::Device;
pub type MyAutodiffBackend = Autodiff<MyBackend>;