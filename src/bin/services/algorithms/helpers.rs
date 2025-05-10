use std::fmt::Display;
use std::io;
use burn::module::AutodiffModule;
use burn::prelude::*;
use rand::prelude::IteratorRandom;
use rand::Rng;
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::config::{MyAutodiffBackend, MyBackend};
use crate::services::algorithms::model::{Forward, MyQmlp};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

pub fn softmax<B: AutodiffBackend>(logits: Tensor<B, 1>) -> Tensor<B, 1> {
    let max_val = logits.clone().max();
    let shifted = logits - max_val.clone();
    let exp    = shifted.exp();
    let sum    = exp.clone().sum();
    exp.div(sum)
}

pub fn masked_softmax<B: AutodiffBackend>(
    logits: Tensor<B, 1>,
    mask:   Tensor<B, 1>,
) -> Tensor<B, 1> {
    let max_val = (logits.clone() * mask.clone()).max();
    let shifted = logits - max_val.clone();
    let exp_all = shifted.exp();
    let exp_masked = exp_all * mask.clone();
    let sum = exp_masked.clone().sum();
    exp_masked.div(sum)
}

pub fn masked_log_softmax<B: AutodiffBackend>(
    logits: Tensor<B, 1>,
    mask:   Tensor<B, 1>,
) -> Tensor<B, 1> {
    let neg_inf = Tensor::from_floats([f32::MIN; 1], &logits.device());
    let masked_logits = logits.clone() * mask.clone() + neg_inf * (mask.clone().mul_scalar(-1.0).add_scalar(1.0));
    log_softmax(masked_logits)
}

pub fn log_softmax<B: AutodiffBackend, const D: usize>(
    logits: Tensor<B, D>,
) -> Tensor<B, D> {
    let max_val = logits.clone().max_dim(D - 1);
    let shifted = logits.clone() - max_val.clone();
    let exp = shifted.exp();
    let sum_exp = exp.clone().sum_dim(D - 1);
    let lse = sum_exp.log() + max_val;
    logits - lse
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
    model:  MyQmlp<MyAutodiffBackend>,
)
where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let valid_model: MyQmlp<MyBackend> = model.valid();
    let mut env = Env::default();
    env.set_against_random();

    // precompute a vector of -INF for masked-out policy
    let fmin_vec  = Tensor::<MyBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS], device);

    loop {
        env.reset();
        while !env.is_game_over() {
            println!("{}", env);

            let s_tensor = Tensor::<MyBackend, 1>::from_floats(env.state_description().as_slice(), device);
            let mask_tensor = Tensor::<MyBackend, 1>::from(env.action_mask()).to_device(device);

            let out = valid_model.forward(s_tensor);

            // ALWAYS slice only the first NUM_ACTIONS entries as policy logits
            let policy_logits = out.clone().slice([0..NUM_ACTIONS]);

            // pick greedily from the policy head
            let a = greedy_policy_action::<MyBackend>(&policy_logits, &mask_tensor, &fmin_vec);

            env.step_from_idx(a);
        }

        println!("{}", env);
        println!("Press Enter to continue or 'quit' to quit");

        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("quit") {
            break;
        }
    }
}

pub fn split_policy_value<B: Backend, const A: usize>(
    out: Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let policy_logits = out.clone().slice([0..A]);
    let value_v      = out.slice([A..A + 1]);
    (policy_logits, value_v)
}

struct Node<const A: usize> {
    visits: usize,
    value: f32,
    children: [Option<usize>; A],
    untried: Vec<usize>,
}

impl<const A: usize> Node<A> {
    fn new(avail: impl Iterator<Item = usize>) -> Self {
        let untried = avail.collect::<Vec<_>>();
        Node {
            visits: 0,
            value: 0.0,
            children: [(); A].map(|_| None),
            untried,
        }
    }
}

pub fn run_mcts_pi<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
    R: Rng + ?Sized,
>(
    root_env: &Env,
    num_sims: usize,
    c: f32,
    rng: &mut R,
) -> [f32; A] {
    // tree and corresponding env states
    let mut tree: Vec<Node<A>> = Vec::new();
    let mut states: Vec<Env> = Vec::new();

    // create root
    tree.push(Node::new(root_env.available_actions_ids()));
    states.push(root_env.clone());
    let root_id = 0;

    for _ in 0..num_sims {
        let mut node = root_id;
        let mut env = root_env.clone();
        let mut path = vec![node];

        while tree[node].untried.is_empty() && !env.is_game_over() {
            let parent_n = tree[node].visits as f32;
            let (a, &child_opt) = tree[node]
                .children
                .iter()
                .enumerate()
                .filter(|&(_, &opt)| opt.is_some())
                .max_by(|&(a1, _), &(a2, _)| {
                    let c1 = tree[node].children[a1].unwrap();
                    let c2 = tree[node].children[a2].unwrap();
                    let q1 = tree[c1].value / tree[c1].visits as f32;
                    let q2 = tree[c2].value / tree[c2].visits as f32;
                    let u1 = q1 + c * (parent_n.ln() / tree[c1].visits as f32).sqrt();
                    let u2 = q2 + c * (parent_n.ln() / tree[c2].visits as f32).sqrt();
                    u1.partial_cmp(&u2).unwrap()
                })
                .unwrap();
            let child = child_opt.unwrap();
            env.step_from_idx(a);
            node = child;
            path.push(node);
        }

        if !env.is_game_over() {
            let a_mcts = tree[node].untried.pop().unwrap();
            env.step_from_idx(a_mcts);
            let new_id = tree.len();
            tree.push(Node::new(env.available_actions_ids()));
            states.push(env.clone());
            tree[node].children[a_mcts] = Some(new_id);
            node = new_id;
            path.push(node);
        }

        let mut rollout = env.clone();
        while !rollout.is_game_over() {
            let a = rollout.available_actions_ids().choose(rng).unwrap();
            rollout.step_from_idx(a);
        }
        let reward = rollout.score();

        for &n in &path {
            tree[n].visits += 1;
            tree[n].value += reward;
        }
    }

    let mut pi = [0.0f32; A];
    let root = &tree[root_id];
    let total_visits: usize = root
        .children
        .iter()
        .filter_map(|&c| c)
        .map(|ci| tree[ci].visits)
        .sum();
    if total_visits > 0 {
        for (a, &child_opt) in root.children.iter().enumerate() {
            if let Some(ci) = child_opt {
                pi[a] = (tree[ci].visits as f32) / (total_visits as f32);
            }
        }
    }
    pi
}

pub fn greedy_policy_action<B: Backend<FloatElem = f32, IntElem = i64>>(
    policy_logits:  &Tensor<B, 1>,
    mask_tensor:    &Tensor<B, 1>,
    fmin_vec:       &Tensor<B, 1>,
) -> usize {
    let masked = policy_logits.clone() * mask_tensor.clone()
        + (mask_tensor.clone().mul_scalar(-1.0).add_scalar(1.0)) * fmin_vec.clone();
    masked.argmax(0).into_scalar() as usize
}