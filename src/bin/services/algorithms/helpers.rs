use rand_distr::Distribution;
use std::fmt::Display;
use std::io;
use std::path::Path;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rand::prelude::IteratorRandom;
use rand::Rng;
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::config::{MyAutodiffBackend, MyBackend};
use crate::services::algorithms::model::{Forward, MyQmlp};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::module::Module;


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

struct Edge {
    action: usize,
    child : usize,
}

struct Node<const S: usize, const A: usize> {
    visits: usize,
    value: f32,
    children: [Option<Edge>; A],
    tried: Vec<usize>,  // Keep track of actions we've already tried
}

impl<const S: usize, const A: usize> Node<S, A> {
    fn new() -> Self {
        Node {
            visits: 0,
            value: 0.0,
            children: [(); A].map(|_| None),
            tried: Vec::new(),
        }
    }

    // Get untried actions from current legal moves
    fn get_untried_action(&mut self, env: &impl DeepDiscreteActionsEnv<S, A>) -> Option<usize> {
        let legal_moves: Vec<_> = env.available_actions_ids()
            .filter(|&action| !self.tried.contains(&action))
            .collect();

        if let Some(&action) = legal_moves.first() {
            self.tried.push(action);
            Some(action)
        } else {
            None
        }
    }
}


pub fn run_mcts_pi<const S: usize, const A: usize, Env, R>(
    root_env: &Env,
    num_sims: usize,
    c: f32,
    rng: &mut R,
) -> [f32; A]
where
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
    R: Rng + ?Sized,
{
    // We create a new tree
    let mut tree = Vec::new();
    
    // Add the root node 
    tree.push(Node::new());  // Create root node

    for _ in 0..num_sims {
        let mut node = 0;
        let mut env = root_env.clone();
        let mut path = vec![node];

        // Selection
        while !env.is_game_over() {
            // If we have an untried action we expand
            if let Some(action) = tree[node].get_untried_action(&env) {
                // Expansion
                env.step_from_idx(action);
                let new_id = tree.len();
                tree.push(Node::new());
                
                // Add the edge to the parent
                tree[node].children[action] = Some(Edge { action, child: new_id });
                
                // Get new parent
                node = new_id;
                path.push(node);
                break;
            } else {
                // Continue selection
                let parent_n = tree[node].visits as f32;
                let current_legal_moves: Vec<_> = env.available_actions_ids().collect();

                let (_best_edge_idx, best_edge) = tree[node].children
                    .iter()
                    .enumerate()
                    .filter_map(|(a, e)| e.as_ref().map(|edge| (a, edge)))
                    // Only consider edges that correspond to currently legal moves
                    .filter(|(_, edge)| current_legal_moves.contains(&edge.action))
                    .max_by(|&(_, e1), &(_, e2)| {
                        let q1 = tree[e1.child].value / tree[e1.child].visits as f32;
                        let q2 = tree[e2.child].value / tree[e2.child].visits as f32;
                        let u1 = q1 + c * (parent_n.ln() / tree[e1.child].visits as f32).sqrt();
                        let u2 = q2 + c * (parent_n.ln() / tree[e2.child].visits as f32).sqrt();
                        u1.partial_cmp(&u2).unwrap()
                    })
                    .ok_or("No legal moves").unwrap();

                env.step_from_idx(best_edge.action);
                node = best_edge.child;
                path.push(node);
            }
        }

        // Simulation
        let mut rollout = env.clone();
        while !rollout.is_game_over() {
            if let Some(a) = rollout.available_actions_ids().choose(rng) {
                rollout.step_from_idx(a);
            } else {
                break;
            }
        }
        let reward = rollout.score();

        // Backpropagation
        for &n in &path {
            tree[n].visits += 1;
            tree[n].value += reward;
        }
    }

    // Calculate policy from visit counts
    let mut pi = [0.0f32; A];
    let root = &tree[0];
    let total_visits: usize = root.children
        .iter()
        .filter_map(|edge_opt| edge_opt.as_ref().map(|e| e.child))
        .map(|cid| tree[cid].visits)
        .sum();

    if total_visits > 0 {
        for (a, edge_opt) in root.children.iter().enumerate() {
            if let Some(edge) = edge_opt {
                pi[a] = tree[edge.child].visits as f32 / total_visits as f32;
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

// helper ────────────────────────────────────────────────────────────────
pub fn sample_distinct_weighted(
    probs: &[f32],
    batch: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    use rand::distributions::WeightedIndex;
    use std::collections::HashSet;

    let dist = WeightedIndex::new(probs).unwrap();
    let mut set = HashSet::with_capacity(batch);
    while set.len() < batch {
        set.insert(dist.sample(rng));
    }
    set.into_iter().collect()
}

#[allow(dead_code)]
pub fn step_with_model<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M,
    B,
    Env,
>(
    env:    &mut Env,
    model:  &M,
    action: usize,
    device: &B::Device,
)
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS>,
{
    env.step(action);

    if !env.is_game_over() {
        env.switch_board();
        let state = env.state_description();
        let state_tensor = Tensor::<B::InnerBackend, 1>::from_floats(state.as_slice(), device);

        let mask = env.action_mask();
        let mask_tensor = Tensor::<B::InnerBackend, 1>::from_floats(mask.as_slice(), device);
        let output = model.forward(state_tensor);

        let masked_output = output * mask_tensor.clone() +
            (mask_tensor.mul_scalar(-1.0).add_scalar(1.0)) *
                Tensor::<B::InnerBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS], device);

        let model_action = masked_output.argmax(0).into_scalar() as usize;
        env.step_from_idx(model_action);
        env.switch_board();
    }
}

#[allow(dead_code)]
pub fn load_inference_model<M, B>(
    model: M,                 // an *empty* instance of the net
    file_path: impl AsRef<Path>,  // e.g. "model_42.mpk"
    device: &B::Device,
) -> <M as AutodiffModule<B>>::InnerModule
where
    M: Module<B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let loaded = model
        .load_file(file_path.as_ref(), &recorder, device)
        .expect("failed to load checkpoint");

    // drop the AD graph → inner backend, ready for inference
    loaded.valid().clone()
}