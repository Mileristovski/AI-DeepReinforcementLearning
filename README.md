# Deep Reinforcement Learning in Rust

## Overview

This project implements a collection of **Deep Reinforcement Learning (DRL)** algorithms in Rust.  
It provides a modular architecture for training, evaluating, and comparing various model-based and model-free methods across several custom environments.

---

## Features

### ✅ Implemented Algorithms

**Model-Based:**
- Expert Apprentice
- Monte Carlo

**Model-Free:**
- Proximal Policy Optimization (PPO)
- Q-Learning
- REINFORCE
- SARSA

### 🌍 Environments
- Line World
- Grid World
- Tic-Tac-Toe
- Bobail (African traditional game)

---

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (Stable)
- [CUDA](https://developer.nvidia.com/cuda-downloads) *(Optional, for GPU acceleration)*
- Ensure `libtorch` environment variables are properly configured if using CUDA

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/deep-rl-rust.git
   cd deep-rl-rust
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

---

## Usage

Run the project with:
```bash
cargo run --release
```

Configure algorithm/environment via code or future CLI options.

---

## Contributions

Contributions are welcome!  
Feel free to fork the repo, suggest improvements, or submit a pull request.

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

🚀 Happy Reinforcement Learning!
