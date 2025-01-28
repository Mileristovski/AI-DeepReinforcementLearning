mod algorithms;
mod environments;
mod services;
mod gui;

fn main() {
    algorithms::q_learning::double_deep_q_learning::run();
}
