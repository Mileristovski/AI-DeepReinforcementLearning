use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::config::{DeepLearningParams, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algo_helper::helpers::get_device;

struct Node<const A: usize> {
    visits: usize,
    value: f32,
    children: [Option<usize>; A],
    untried: Vec<usize>,
}

impl<const A: usize> Node<A> {
    fn new(available: impl Iterator<Item=usize>) -> Self {
        let untried = available.collect::<Vec<_>>();
        Node {
            visits: 0,
            value: 0.0,
            children: [(); A].map(|_| None),
            untried,
        }
    }
}

pub fn run_mcts<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
>(
    env_name: &str,
) {
    let params = DeepLearningParams::default();
    let mut env = Env::default();
    let _device: MyDevice = get_device();
    println!("Running MCTS on {} with {} sims, c={}", env_name, params.mcts_simulations, params.mcts_c);

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    for ep in tqdm!(0..params.num_episodes) {
        if ep > 0 && ep % params.episode_stop == 0 {
            println!("Episode {}/{} complete", ep, params.num_episodes);
        }
        env.reset();
        env.set_against_random();

        while !env.is_game_over() {
            // perform MCTS from current root state
            let action = mcts_search(&env, params.mcts_simulations, params.mcts_c, &mut rng);
            env.step_from_idx(action);
        }
        println!("{}", env);
    }
}

/// Perform MCTS from the given root‐environment, return the chosen action.
pub fn mcts_search<const S: usize, const A: usize, Env, R>(
    root_env: &Env,
    num_sim: usize,
    c: f32,
    rng: &mut R,
) -> usize
where
    Env: DeepDiscreteActionsEnv<S, A> + Clone,
    R: rand::Rng + ?Sized,
{
    // tree of nodes
    let mut tree: Vec<Node<A>> = Vec::new();
    // map each node to its env state (we only need to clone on expansion)
    let mut states: Vec<Env> = Vec::new();

    // create root
    let root_id = 0;
    tree.push(Node::new(root_env.available_actions_ids()));
    states.push(root_env.clone());

    for _ in 0..num_sim {
        // 1) Selection
        let mut node = root_id;
        let mut env = root_env.clone();
        let mut path = vec![node];

        // descend until we find a node with untried actions or terminal
        while tree[node].untried.is_empty() && !env.is_game_over() {
            // UCT select
            let parent_n = tree[node].visits as f32;
            let (best_a, &child_opt) = tree[node].children.iter()
                .enumerate()
                .filter(|&(a, &c)| c.is_some())
                .max_by(|&(a,_),&(b,_)| {
                    let ca = tree[node].children[a].unwrap();
                    let cb = tree[node].children[b].unwrap();
                    let qa = tree[ca].value / tree[ca].visits as f32;
                    let qb = tree[cb].value / tree[cb].visits as f32;
                    let ua = qa + c * (parent_n.ln() / tree[ca].visits as f32).sqrt();
                    let ub = qb + c * (parent_n.ln() / tree[cb].visits as f32).sqrt();
                    ua.partial_cmp(&ub).unwrap()
                }).unwrap();
            let child = child_opt.unwrap();
            env.step_from_idx(best_a);
            node = child;
            path.push(node);
        }

        // 2) Expansion
        if !env.is_game_over() {
            // pick one untried action
            let a = tree[node].untried.pop().unwrap();
            env.step_from_idx(a);
            // create new child
            let new_id = tree.len();
            tree.push(Node::new(env.available_actions_ids()));
            states.push(env.clone());
            tree[node].children[a] = Some(new_id);
            node = new_id;
            path.push(node);
        }

        // 3) Simulation (roll‑out)
        let mut rollout_env = env.clone();
        while !rollout_env.is_game_over() {
            let a = rollout_env.available_actions_ids().choose(rng).unwrap();
            rollout_env.step_from_idx(a);
        }
        let reward = rollout_env.score();

        // 4) Backpropagate
        for &n in &path {
            tree[n].visits += 1;
            tree[n].value += reward;
        }
    }

    // at root, pick the action with highest visit count
    let root = &tree[root_id];
    root.children.iter()
        .enumerate()
        .filter(|&(_, &c)| c.is_some())
        .max_by_key(|&(a, &c)| {
            let ci = c.unwrap();
            tree[ci].visits
        })
        .map(|(a, _)| a)
        .unwrap()
}