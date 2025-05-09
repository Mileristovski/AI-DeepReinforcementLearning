use std::fmt::Display;
use std::io;
use burn::module::AutodiffModule;
use burn::prelude::*;
use rand::prelude::IteratorRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::config::{MyAutodiffBackend, MyBackend};
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};

pub fn argmax(row: &Vec<f32>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap()
        .0
}


pub fn max(row: &Vec<f32>) -> f32 {
    *row.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}


pub fn epsilon_greedy_action<B: Backend<FloatElem=f32, IntElem=i64>, const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    minus_one: &Tensor<B, 1>,
    plus_one:  &Tensor<B, 1>,
    fmin_vec:  &Tensor<B, 1>,
    available_actions: impl Iterator<Item=usize>,
    epsilon: f32,
    rng: &mut impl Rng
) -> usize {
    if rng.gen_range(0f32..=1f32) < epsilon {
        available_actions.choose(rng).unwrap()
    } else {
        let inverted_mask = mask_tensor.clone() * minus_one.clone() + plus_one.clone();
        let masked_q_s    = q_s.clone() * mask_tensor.clone()
            + inverted_mask * fmin_vec.clone();

        masked_q_s.argmax(0).into_scalar() as usize
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

pub fn test_trained_model<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    device: &<MyBackend as Backend>::Device,
    model: MyQmlp<MyAutodiffBackend>,
)
where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let valid_model: MyQmlp<MyBackend> = model.valid();
    let mut env = Env::default();
    env.set_against_random();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let minus_one = Tensor::<MyBackend, 1>::from_floats([-1.0; NUM_ACTIONS], device);
    let plus_one  = Tensor::<MyBackend, 1>::from_floats([ 1.0; NUM_ACTIONS], device);
    let fmin_vec  = Tensor::<MyBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS], device);

    loop {
        env.reset();
        while !env.is_game_over() {
            println!("{}", env);

            let s_tensor = Tensor::<MyBackend, 1>::from_floats(env.state_description().as_slice(), device);
            let mask_tensor = Tensor::<MyBackend, 1>::from(env.action_mask()).to_device(device);
            
            let q_s: Tensor<MyBackend, 1> = valid_model.forward(s_tensor);

            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s,
                &mask_tensor,
                &minus_one,
                &plus_one,
                &fmin_vec,
                env.available_actions_ids(),
                1e-5,
                &mut rng,
            );
            env.step_from_idx(a);
        }

        println!("{}", env);
        println!("Press Enter to continue or 'quit' to quit");
        
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("quit")  {  
            break 
        }
    }
}