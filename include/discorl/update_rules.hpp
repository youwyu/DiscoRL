#pragma once

#include <functional>
#include <utility>

#include <torch/torch.h>

#include "discorl/networks.hpp"
#include "discorl/params.hpp"
#include "discorl/value.hpp"

namespace discorl {

// A replay window as an update rule sees it (types.UpdateRuleInputs).
struct RuleInputs {
  torch::Tensor observations; // [T+1, B, D]
  torch::Tensor actions;      // [T+1, B]
  torch::Tensor rewards;      // [T, B]
  torch::Tensor is_terminal;  // [T, B]
  Tensors agent_out;          // [T+1, B, ...] with the current parameters
  Tensors behaviour_out;      // [T+1, B, ...] recorded when acting
};

// What a rule carries between updates.
struct MetaState {
  LstmState rnn; // Disco: the lifetime meta-RNN
  Ema adv_ema;
  Ema td_ema;
  Params target_params; // Disco: the target network
};

// Agent outputs for any parameters, [T+1, B, ...] from observations.
using UnrollFn = std::function<Tensors(const Params &, const torch::Tensor &)>;

// update_rules/base.py.
class UpdateRule {
public:
  virtual ~UpdateRule() = default;

  virtual OutputSpec output_spec(int64_t num_actions) const = 0;
  // Meta-parameters; empty without a meta-network.
  virtual Params init_params(const torch::TensorOptions &options) const;
  virtual MetaState init_meta_state(const Params &agent_params) const = 0;
  // The loss targets of a window, and the rule's next state.
  virtual std::pair<Tensors, MetaState>
  unroll_meta_net(const Params &meta_params, const Params &agent_params,
                  const MetaState &state, const RuleInputs &inputs,
                  const UnrollFn &unroll) const = 0;
  // Per-step loss [T, B]; backprop keeps the targets' gradients.
  virtual torch::Tensor agent_loss(const RuleInputs &inputs,
                                   const Tensors &meta_out,
                                   bool backprop) const = 0;
  // Per-step loss [T, B] without meta-gradients.
  virtual torch::Tensor agent_loss_no_meta(const RuleInputs &inputs,
                                           const Tensors &meta_out) const;
};

// DiscoRL's discovered rule (update_rules/disco.py), settings of
// agent.get_settings_disco.
struct DiscoConfig {
  MetaNetConfig net;
  double value_discount = 0.997;
  double max_abs_value = 300.0;
  int64_t num_bins = 601;
  double ema_decay = 0.99;
  double ema_eps = 1e-6;
  double pi_cost = 1.0;
  double y_cost = 1.0;
  double z_cost = 1.0;
  double aux_policy_cost = 1.0;
  double value_cost = 0.2;
  double target_params_coeff = 0.9;
  double td_lambda = 0.95;
};

class DiscoRule : public UpdateRule {
public:
  explicit DiscoRule(DiscoConfig config = {});

  OutputSpec output_spec(int64_t num_actions) const override;
  Params init_params(const torch::TensorOptions &options) const override;
  MetaState init_meta_state(const Params &agent_params) const override;
  std::pair<Tensors, MetaState>
  unroll_meta_net(const Params &meta_params, const Params &agent_params,
                  const MetaState &state, const RuleInputs &inputs,
                  const UnrollFn &unroll) const override;
  torch::Tensor agent_loss(const RuleInputs &inputs, const Tensors &meta_out,
                           bool backprop) const override;
  torch::Tensor agent_loss_no_meta(const RuleInputs &inputs,
                                   const Tensors &meta_out) const override;

private:
  ValueOptions value_options() const;

  DiscoConfig config_;
  MetaNet net_;
};

// Actor-critic with V-trace (update_rules/actor_critic.py), settings of
// agent.get_settings_actor_critic.
struct ActorCriticConfig {
  bool categorical_value = true;
  int64_t num_bins = 601;
  double max_abs_value = 300.0;
  bool nonlinear_value_transform = true;
  bool normalize_adv = false;
  bool normalize_td = false;
  double ema_decay = 0.99;
  double ema_eps = 1e-6;
  double discount = 0.997;
  double vtrace_lambda = 0.95;
  double entropy_cost = 0.2;
  double pg_cost = 1.0;
  double value_cost = 0.5;
};

class ActorCritic : public UpdateRule {
public:
  explicit ActorCritic(ActorCriticConfig config = {});

  OutputSpec output_spec(int64_t num_actions) const override;
  MetaState init_meta_state(const Params &agent_params) const override;
  std::pair<Tensors, MetaState>
  unroll_meta_net(const Params &meta_params, const Params &agent_params,
                  const MetaState &state, const RuleInputs &inputs,
                  const UnrollFn &unroll) const override;
  torch::Tensor agent_loss(const RuleInputs &inputs, const Tensors &meta_out,
                           bool backprop) const override;

private:
  ValueOptions value_options() const;

  ActorCriticConfig config_;
};

} // namespace discorl
