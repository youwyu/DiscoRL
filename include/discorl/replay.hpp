#pragma once

#include <deque>
#include <random>

#include "discorl/agent.hpp"

namespace discorl {

// A FIFO of single trajectories, sampled uniformly (the SimpleReplayBuffer
// of colabs/eval.ipynb).
class ReplayBuffer {
public:
  explicit ReplayBuffer(size_t capacity, uint64_t seed = 0);

  // Adds each of the rollout's B trajectories.
  void add(const Rollout &rollout);
  // B trajectories, with replacement.
  Rollout sample(int64_t batch_size);
  size_t size() const { return trajectories_.size(); }

private:
  size_t capacity_;
  std::mt19937_64 rng_;
  std::deque<Rollout> trajectories_;
};

} // namespace discorl
