#pragma once

#include <memory>
#include <utility>

#include <torch/torch.h>

#include "discorl/networks.hpp"
#include "discorl/params.hpp"
#include "discorl/update_rules.hpp"
#include "discorl/value.hpp"

namespace discorl {

// Acting steps [T, B, ...] (types.ActorRollout): step t holds the observation,
// the action taken in it, and the reward and discount of arriving there.
struct Rollout {
  torch::Tensor observations; // [T, B, D]
  torch::Tensor actions;      // [T, B]
  torch::Tensor rewards;      // [T, B]
  torch::Tensor discounts;    // [T, B], 0 where an episode ended
  Tensors agent_outs;         // [T, B, ...] of the acting network: "logits"

  Rollout slice(int64_t batch_begin, int64_t batch_end) const;
  Rollout to(const torch::Device &device) const;
  static Rollout cat(const std::vector<Rollout> &rollouts); // along batch
};

// optax.chain(scale_by_adam_sg_denom(), clip(max_abs_update), scale(-lr)):
// Adam whose denominator gets no meta-gradient, then per-element clipping.
struct AdamState {
  int64_t count = 0;
  Params mu;
  Params nu;
};

struct Optimizer {
  double learning_rate = 3e-4;
  double max_abs_update = 1.0;
  double b1 = 0.9;
  double b2 = 0.999;
  double eps = 1e-8;

  AdamState init(const Params &params) const;
  // The updated parameters and state.
  std::pair<Params, AdamState> step(const Params &params, const Params &grads,
                                    const AdamState &state) const;
};

struct AgentConfig {
  NetConfig net;
  double learning_rate = 3e-4;
  double max_abs_update = 1.0;
};

struct LearnerState {
  Params params;
  AdamState opt;
  MetaState meta;
};

// An agent learning by an update rule (agent.py).
class Agent {
public:
  Agent(AgentConfig config, std::shared_ptr<const UpdateRule> rule,
        int64_t observation_size, int64_t num_actions);

  LearnerState initial_learner_state(const torch::TensorOptions &options) const;
  Tensors unroll(const Params &params, const torch::Tensor &observations) const;
  // Actions sampled from the policy for observations [B, D], and its logits.
  std::pair<torch::Tensor, torch::Tensor>
  act(const Params &params, const torch::Tensor &observation) const;
  // One update of the rollout's [T, B] steps. When meta-training, the new
  // state keeps its graph to the meta-parameters. The logs hold "total_loss"
  // and the rule's targets under "meta_out/".
  std::pair<LearnerState, Tensors> learner_step(const Rollout &rollout,
                                                const LearnerState &state,
                                                const Params &meta_params,
                                                bool meta_training) const;

  const UpdateRule &rule() const { return *rule_; }
  int64_t num_actions() const { return num_actions_; }

private:
  torch::Tensor loss(const Params &params, const RuleInputs &inputs,
                     const torch::Tensor &discounts, const Tensors &meta_out,
                     bool backprop) const;

  std::shared_ptr<const UpdateRule> rule_;
  AgentNet net_;
  Optimizer optimizer_;
  int64_t observation_size_;
  int64_t num_actions_;
};

// The state-value function meta-training uses for advantages
// (value_fns/value_fn.py).
struct ValueFnConfig {
  std::vector<int64_t> dense{256, 256};
  double head_w_init_std = 1e-2;
  double learning_rate = 1e-3;
  double max_abs_update = 1.0;
  double discount = 0.99;
  double td_lambda = 0.96;
  double outer_value_cost = 1.0;
  double ema_decay = 0.99;
  double ema_eps = 1e-6;
};

struct ValueState {
  Params params;
  AdamState opt;
  Ema adv_ema;
  Ema td_ema;
};

class ValueFunction {
public:
  ValueFunction(ValueFnConfig config, int64_t observation_size);

  ValueState initial_state(const torch::TensorOptions &options) const;
  // V-trace estimates for the policy logits [T, B, A]; with its EMAs.
  std::pair<ValueOuts, ValueState>
  value_outs(const ValueState &state, const Rollout &rollout,
             const torch::Tensor &logits) const;
  ValueState update(const ValueState &state, const Rollout &rollout,
                    const torch::Tensor &logits) const;

private:
  ValueFnConfig config_;
  AgentNet net_;
  Optimizer optimizer_;
  int64_t observation_size_;
};

} // namespace discorl
