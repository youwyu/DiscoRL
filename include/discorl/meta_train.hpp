#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "discorl/agent.hpp"
#include "discorl/env.hpp"
#include "discorl/update_rules.hpp"

namespace discorl {

// Meta-training of the discovered rule, as colabs/meta_train.ipynb: each
// agent updates with the rule on training rollouts, and the rule follows the
// meta-gradient of the updated agent's policy-gradient loss on a validation
// rollout.
struct MetaTrainConfig {
  int64_t num_agents = 2;
  int64_t rollout_len = 16;
  int64_t num_inner_steps = 2;
  int64_t batch_size = 32;
  double learning_rate = 5e-4; // Adam on the meta-parameters
  // Meta-loss regularizers.
  double entropy_cost = 1e-2;            // the validation policy's entropy
  double prediction_entropy_cost = 1e-3; // y and z entropies
  double target_mean_cost = 1e-3;        // squared means of the targets
  double target_kl_cost = 1e-2;          // KL(target policy || pi target)
};

struct MetaGradient {
  torch::Tensor loss;
  Params grads;
  LearnerState learner; // after the inner updates, without history
  ValueState value;
  Tensors logs;
};

// The meta-gradient of one agent: inner updates on `train`, each rollout
// [L, B], then the meta-loss on `valid`, [2L, B], acted before the updates.
MetaGradient meta_gradient(const Agent &agent, const ValueFunction &value_fn,
                           const Params &meta_params,
                           const LearnerState &learner, const ValueState &value,
                           const std::vector<Rollout> &train,
                           const Rollout &valid, const MetaTrainConfig &config);

using EnvironmentFactory = std::function<std::unique_ptr<Environment>(
    int64_t batch_size, uint64_t seed)>;

class MetaTrainer {
public:
  MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
              ValueFnConfig value_config, DiscoConfig rule_config,
              const EnvironmentFactory &make_env, Params meta_params,
              const torch::TensorOptions &options, uint64_t seed = 0);

  // One meta-update across the population.
  Tensors step();
  const Params &meta_params() const { return meta_params_; }

private:
  struct Member {
    std::unique_ptr<Environment> env;
    TimeStep timestep;
    LearnerState learner;
    ValueState value;
  };

  MetaTrainConfig config_;
  std::unique_ptr<Agent> agent_;
  std::unique_ptr<ValueFunction> value_fn_;
  Params meta_params_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
  std::vector<Member> members_;
};

} // namespace discorl
