#include "discorl/replay.hpp"

#include <vector>

namespace discorl {

ReplayBuffer::ReplayBuffer(size_t capacity, uint64_t seed)
    : capacity_(capacity), rng_(seed) {}

void ReplayBuffer::add(const Rollout &rollout) {
  for (int64_t b = 0; b < rollout.actions.size(1); ++b) {
    trajectories_.push_back(rollout.slice(b, b + 1));
    if (trajectories_.size() > capacity_) {
      trajectories_.pop_front();
    }
  }
}

Rollout ReplayBuffer::sample(int64_t batch_size) {
  std::uniform_int_distribution<size_t> index(0, trajectories_.size() - 1);
  std::vector<Rollout> batch;
  batch.reserve(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    batch.push_back(trajectories_[index(rng_)]);
  }
  return Rollout::cat(batch);
}

} // namespace discorl
