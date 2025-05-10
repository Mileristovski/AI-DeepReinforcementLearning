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
                .filter(|&(_, &c)| c.is_some())
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
            let a = tree[node].untried.pop().unwrap();
            env.step_from_idx(a);
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
        .max_by_key(|&(_, &c)| {
            let ci = c.unwrap();
            tree[ci].visits
        })
        .map(|(a, _)| a)
        .unwrap()
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

    for ep in tqdm!(0..params.num_episodes) {
        if ep > 0 && ep % params.episode_stop == 0 {
            println!("Mean Score : {:.3}", total / params.episode_stop as f32);
            total = 0.0;
        }
        env.reset();
        env.set_against_random();

        while !env.is_game_over() {
            let action = mcts_search(&env, params.mcts_simulations, params.mcts_c, rng);
            env.step_from_idx(action);
        }
        total += env.score();
    }
}

/// “Test” episodes in a loop, prompting user to press Enter or “quit”
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
        env.set_against_random();
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

/// entry point: first training, then interactive testing — just like random rollout
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