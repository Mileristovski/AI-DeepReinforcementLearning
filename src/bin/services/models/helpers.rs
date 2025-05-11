use std::fmt::Display;
use burn::module::AutodiffModule;
use burn::prelude::{Backend, Tensor};
use crate::config::{MyAutodiffBackend, MyBackend};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::helpers::greedy_policy_action;
use crate::services::algorithms::model::{Forward, MyQmlp};

pub fn compare_models<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(
    device: &<MyBackend as Backend>::Device,
    model1: MyQmlp<MyAutodiffBackend>,
    model2: MyQmlp<MyAutodiffBackend>,
    num_games: usize,
) -> (f32, f32, f32) // Returns (model1_wins, model2_wins, draws) as percentages
where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let valid_model1: MyQmlp<MyBackend> = model1.valid();
    let valid_model2: MyQmlp<MyBackend> = model2.valid();
    let fmin_vec = Tensor::<MyBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS], device);

    let mut model1_wins = 0f32;
    let mut model2_wins = 0f32;
    let mut draws = 0f32;

    for game in 0..num_games {
        let mut env = Env::default();
        let (first_model, second_model) = if game % 2 == 0 {
            (&valid_model1, &valid_model2)
        } else {
            (&valid_model2, &valid_model1)
        };
        let mut current_player = 0;

        while !env.is_game_over() {
            let current_model = if current_player == 0 { first_model } else { second_model };

            let s_tensor = Tensor::<MyBackend, 1>::from_floats(env.state_description().as_slice(), device);
            let mask_tensor = Tensor::<MyBackend, 1>::from(env.action_mask()).to_device(device);

            let out = current_model.forward(s_tensor);
            let policy_logits = out.clone().slice([0..NUM_ACTIONS]);
            let action = greedy_policy_action::<MyBackend>(&policy_logits, &mask_tensor, &fmin_vec);

            env.step_from_idx(action);
            env.switch_board();
            current_player = if current_player == 0 { 1 } else { 0 }
        }

        let final_score = env.score();
        if final_score == 0.0 {
            draws += 1.0;
        } else if game % 2 == 0 {
            if final_score > 0.0 { model1_wins += 1.0; } else { model2_wins += 1.0; }
        } else {
            if final_score > 0.0 { model2_wins += 1.0; } else { model1_wins += 1.0; }
        }

        // Print progress
        if (game + 1) % 10 == 0 {
            println!("Completed {} games", game + 1);
            println!("Current stats: Model1 wins: {:.1}%, Model2 wins: {:.1}%, Draws: {:.1}%",
                     100.0 * model1_wins / (game as f32 + 1.0),
                     100.0 * model2_wins / (game as f32 + 1.0),
                     100.0 * draws / (game as f32 + 1.0));
        }
    }

    // Convert to percentages
    let total_games = num_games as f32;
    (
        100.0 * model1_wins / total_games,
        100.0 * model2_wins / total_games,
        100.0 * draws / total_games
    )
}