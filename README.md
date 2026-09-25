<div align="center">

# DiscoRL · C++

**Discover reinforcement-learning algorithms on your own tasks.**

A C++/CUDA port of DeepMind's [DiscoRL](https://www.nature.com/articles/s41586-025-09761-x) (*Nature* 2025):
the meta-learned update rule, and the meta-training that discovers it.

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](#quick-start)
[![LibTorch](https://img.shields.io/badge/LibTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/cppdocs/)
[![Nature 2025](https://img.shields.io/badge/Nature-2025-0f6b99)](https://www.nature.com/articles/s41586-025-09761-x)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results-dark.png">
  <img src="docs/results-light.png" width="100%"
       alt="Left: a rule meta-trained from random parameters teaches its agents Catch, from 20% to 100% of balls caught by meta-update 800. Right: fresh agents trained with it reach an average return of 0.78 in 1000 updates, Disco103's reach 1.0, actor-critic's stay at -0.74.">
</picture>

</div>

## Highlights

- 🧠 **Discover rules from scratch** on any batched environment, racing included
- 🎯 **Faithful**: rule, agent and meta-gradients match DeepMind's [disco_rl](https://github.com/google-deepmind/disco_rl) (JAX) to float64 precision
- 🔁 **Interchangeable**: loads `disco_103.npz`, and saves rules that disco_rl loads
- ⚡ **GPU-native**: LibTorch's CUDA kernels, with the second-order gradients meta-training needs

## How it works

```mermaid
flowchart LR
    env[Environments] -->|experience| agents[Agents]
    rule[Meta-network<br/>the update rule] -->|loss targets| agents
    agents -.->|meta-gradient through their updates| rule
```

## Quick start

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch && cmake --build build -j

build/discorl_meta_train out=rule.npz                 # discover a rule
build/discorl_train rule=disco rule_params=rule.npz   # train an agent with it
```

Flags are `key=value`, for example `env=catch steps=800 device=cuda`; `init=disco_103.npz` fine-tunes DeepMind's rule.

## Your environment

Batched, self-resetting, with discrete actions. Register it in [`apps/common.hpp`](apps/common.hpp).

```cpp
struct Track : discorl::Environment {
  discorl::TimeStep reset() override;
  discorl::TimeStep step(const torch::Tensor &actions) override; // [B] ints
  int64_t batch_size() const override;
  int64_t observation_size() const override;
  int64_t num_actions() const override;
};
```

<details>
<summary><b>Citation</b></summary>

```bibtex
@article{oh2025discovering,
  title   = {Discovering state-of-the-art reinforcement learning algorithms},
  author  = {Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A.
             and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder
             and van Hasselt, Hado and Silver, David},
  journal = {Nature},
  volume  = {648},
  pages   = {312--319},
  year    = {2025},
  doi     = {10.1038/s41586-025-09761-x}
}
```

</details>

Apache 2.0 · a port of [google-deepmind/disco_rl](https://github.com/google-deepmind/disco_rl)
