#pragma once

#include <memory>

#include <torch/torch.h>

#include "discorl/agent.hpp"

namespace discorl {

// A batch of environment steps.
struct TimeStep {
  torch::Tensor observation; // [B, D]
  torch::Tensor reward;      // [B], for arriving at the observation
  torch::Tensor last;        // [B], 1 where an episode ended; the observation
                             // then already starts the next one
};

// A batch of environments that reset themselves, as disco_rl's jittable
// environments. Actions are integers in [0, num_actions()).
class Environment {
public:
  virtual ~Environment() = default;

  virtual TimeStep reset() = 0;
  virtual TimeStep step(const torch::Tensor &actions) = 0;
  virtual int64_t batch_size() const = 0;
  virtual int64_t observation_size() const = 0;
  virtual int64_t num_actions() const = 0;
};

// Catch (environments/jittable_envs.py): a ball falls down a board and the
// paddle on its bottom row moves left, stays or moves right; catching the
// ball gives +1, missing it -1.
class Catch : public Environment {
public:
  Catch(int64_t batch_size, int64_t rows = 8, int64_t columns = 8,
        torch::Device device = torch::kCPU, uint64_t seed = 0);

  TimeStep reset() override;
  TimeStep step(const torch::Tensor &actions) override;
  int64_t batch_size() const override { return batch_size_; }
  int64_t observation_size() const override { return rows_ * columns_; }
  int64_t num_actions() const override { return 3; }

private:
  torch::Tensor new_balls(int64_t n);
  torch::Tensor render() const;

  int64_t batch_size_;
  int64_t rows_;
  int64_t columns_;
  torch::Device device_;
  torch::Generator generator_;
  torch::Tensor ball_y_;   // [B]
  torch::Tensor ball_x_;   // [B]
  torch::Tensor paddle_x_; // [B]
};

// Acts `length` steps from `timestep`, which it advances.
Rollout collect(const Agent &agent, const Params &params, Environment &env,
                TimeStep &timestep, int64_t length);

} // namespace discorl
