use std::fmt::Display;
use std::io;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::config::{DeepLearningParams, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::helpers::get_device;

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
    let mut tree: Vec<Node<A>> = vec![Node::new(root_env.available_actions_ids())];

    for _ in 0..num_sim {
        let mut env = root_env.clone();
        let mut path = Vec::<usize>::with_capacity(64);
        let mut node = 0; // root id
        path.push(node);

        while tree[node].untried.is_empty() && !env.is_game_over() {
            let parent_n = tree[node].visits as f32;
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();
            
            let next = tree[node]
                .children
                .iter()
                .enumerate()
                .filter_map(|(a, &c)| c.map(|id| {
                    if id >= tree.len() {
                        None
                    } else {
                        // Only consider moves that are currently legal
                        if current_legal_moves.contains(&a) {
                            Some((a, id))
                        } else {
                            None
                        }
                    }
                }))
                .flatten()
                .map(|(a, id)| {
                    let q = tree[id].value / tree[id].visits.max(1) as f32;
                    let u = c * ((parent_n.ln() / tree[id].visits.max(1) as f32).sqrt());
                    (q + u, a, id)
                })
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

            match next {
                Some((_, best_a, child)) => {
                    env.step_from_idx(best_a);
                    node = child;
                    path.push(node);
                }
                None => break, // No valid moves available
            }
        }

        // Expansion phase
        if !env.is_game_over() {
            // Get current legal moves
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();
            
            // Find first untried action that is currently legal
            while let Some(a) = tree[node].untried.pop() {
                if current_legal_moves.contains(&a) {
                    env.step_from_idx(a);
                    let id = tree.len();
                    tree.push(Node::new(env.available_actions_ids()));
                    tree[node].children[a] = Some(id);
                    node = id;
                    path.push(node);
                    break;
                }
                // if action is not legal, continue popping until we find a legal one
            }
        }

        // Simulation phase
        while !env.is_game_over() {
            if let Some(a) = env.available_actions_ids().choose(rng) {
                env.step_from_idx(a);
            } else {
                break;
            }
        }

        // Backpropagation phase
        let reward = env.score();
        for &n in &path {
            tree[n].visits += 1;
            tree[n].value += reward;
        }
    }

    // Return best action at root
    tree[0]
        .children
        .iter()
        .enumerate()
        .filter_map(|(a, &c)| {
            // Only consider actions that are currently available
            if root_env.action_mask()[a] == 0.0 {
                None
            } else {
                c.map(|id| (a, tree[id].visits))
            }
        })
        .max_by_key(|&(_, v)| v)
        .map(|(a, _)| a)
        .unwrap_or_else(|| root_env.available_actions_ids().next().unwrap())
}

fn episodic_mcts<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
>(
    params: &DeepLearningParams,
    env_name: &str,
    rng: &mut Xoshiro256PlusPlus,
) where
    [(); S]:,
    [(); A]:,
{
    println!("Running MCTS on {} with {} sims, c={}", env_name, params.mcts_simulations, params.mcts_c);

    let mut env = Env::default();
    let mut total = 0.0;

    for ep in tqdm!(0..=params.num_episodes) {
        if ep > 0 && ep % params.episode_stop == 0 {
            println!("Mean Score : {:.3}", total / params.episode_stop as f32);
            total = 0.0;
        }
        env.reset();

        while !env.is_game_over() {
            let action = mcts_search(&env, params.mcts_simulations, params.mcts_c, rng);
            env.step_from_idx(action);
        }
        total += env.score();
    }
}

fn test_mcts<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
>(
    params: &DeepLearningParams,
    rng: &mut Xoshiro256PlusPlus,
) where
    [(); S]:,
    [(); A]:,
{
    loop {
        let mut env = Env::default();
        env.reset();

        println!("\n--- Test Episode ---\n");
        while !env.is_game_over() {
            let action = mcts_search(&env, params.mcts_simulations, params.mcts_c, rng);
            env.step_from_idx(action);
        }
        let score = env.score();
        println!("\nFinal state:\n{}\nScore: {}", env, score);
        println!("Press Enter to run another test, or type 'quit' to return.");
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("quit") {
            break;
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
    let _device: MyDevice = get_device();
    println!("Using device: {:?}", _device);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.rng_seed);

    episodic_mcts::<S, A, Env>(&params, env_name, &mut rng);
    test_mcts::<S, A, Env>(&params, &mut rng);
}