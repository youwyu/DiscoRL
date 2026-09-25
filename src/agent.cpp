#include "discorl/agent.hpp"

#include <cmath>
#include <utility>

namespace discorl {
namespace {

torch::Tensor drop_last(const torch::Tensor &x) {
  return x.narrow(0, 0, x.size(0) - 1);
}

Ema detached(const Ema &ema) {
  return {ema.moment1.detach(), ema.moment2.detach(),
          ema.decay_product.detach()};
}

} // namespace

Rollout Rollout::slice(int64_t batch_begin, int64_t batch_end) const {
  const auto part = [&](const torch::Tensor &x) {
    return x.narrow(1, batch_begin, batch_end - batch_begin);
  };
  Rollout out{
      part(observations), part(actions), part(rewards), part(discounts), {}};
  for (const auto &[name, value] : agent_outs) {
    out.agent_outs[name] = part(value);
  }
  return out;
}

Rollout Rollout::cat(const std::vector<Rollout> &rollouts) {
  const auto join = [&](auto field) {
    std::vector<torch::Tensor> parts;
    for (const auto &r : rollouts) {
      parts.push_back(field(r));
    }
    return torch::cat(parts, 1);
  };
  Rollout out{join([](const Rollout &r) { return r.observations; }),
              join([](const Rollout &r) { return r.actions; }),
              join([](const Rollout &r) { return r.rewards; }),
              join([](const Rollout &r) { return r.discounts; }),
              {}};
  for (const auto &[name, value] : rollouts.front().agent_outs) {
    out.agent_outs[name] = join(
        [&name = name](const Rollout &r) { return r.agent_outs.at(name); });
  }
  return out;
}

AdamState Optimizer::init(const Params &params) const {
  AdamState state;
  for (const auto &[name, value] : params) {
    state.mu[name] = torch::zeros_like(value).detach();
    state.nu[name] = torch::zeros_like(value).detach();
  }
  return state;
}

std::pair<Params, AdamState> Optimizer::step(const Params &params,
                                             const Params &grads,
                                             const AdamState &state) const {
  AdamState next;
  next.count = state.count + 1;
  const double debias1 = 1.0 - std::pow(b1, next.count);
  const double debias2 = 1.0 - std::pow(b2, next.count);
  Params updated;
  for (const auto &[name, grad] : grads) {
    const auto mu = b1 * state.mu.at(name) + (1.0 - b1) * grad;
    const auto nu = b2 * state.nu.at(name) + (1.0 - b2) * grad.square();
    const auto update =
        (mu / debias1) / (torch::sqrt((nu / debias2).detach()) + eps);
    updated[name] =
        params.at(name) -
        learning_rate * update.clamp(-max_abs_update, max_abs_update);
    next.mu[name] = mu;
    next.nu[name] = nu;
  }
  return {updated, next};
}

Agent::Agent(AgentConfig config, std::shared_ptr<const UpdateRule> rule,
             int64_t observation_size, int64_t num_actions)
    : rule_(std::move(rule)),
      net_(config.net, rule_->output_spec(num_actions), num_actions),
      optimizer_{config.learning_rate, config.max_abs_update},
      observation_size_(observation_size), num_actions_(num_actions) {}

LearnerState
Agent::initial_learner_state(const torch::TensorOptions &options) const {
  auto params = net_.init(observation_size_, options);
  auto meta = rule_->init_meta_state(params);
  return {params, optimizer_.init(params), meta};
}

Tensors Agent::unroll(const Params &params,
                      const torch::Tensor &observations) const {
  return net_(params, observations);
}

std::pair<torch::Tensor, torch::Tensor>
Agent::act(const Params &params, const torch::Tensor &observation) const {
  torch::NoGradGuard no_grad;
  auto logits = net_(params, observation, /*model=*/false).at("logits");
  auto actions = torch::multinomial(logits.softmax(-1), 1).squeeze(-1);
  return {actions, logits};
}

torch::Tensor Agent::loss(const Params &params, const RuleInputs &like,
                          const torch::Tensor &discounts,
                          const Tensors &meta_out, bool backprop) const {
  RuleInputs inputs = like;
  inputs.agent_out = unroll(params, inputs.observations);
  const auto per_step = rule_->agent_loss(inputs, meta_out, backprop) +
                        rule_->agent_loss_no_meta(inputs, meta_out);
  // Steps from an episode's last observation are not transitions.
  const auto masks = (drop_last(discounts) > 0).to(per_step.options());
  return (per_step * masks).sum() / (masks.sum() + 1e-8);
}

std::pair<LearnerState, Tensors> Agent::learner_step(const Rollout &rollout,
                                                     const LearnerState &state,
                                                     const Params &meta_params,
                                                     bool meta_training) const {
  torch::AutoGradMode rule_grad(meta_training);
  auto params = state.params;
  if (meta_training) {
    for (auto &[name, value] : params) {
      if (!value.requires_grad()) {
        value = value.detach().requires_grad_(true);
      }
    }
  }

  const int64_t t_len = rollout.rewards.size(0) - 1;
  RuleInputs inputs;
  inputs.observations = rollout.observations;
  inputs.actions = rollout.actions;
  inputs.rewards = rollout.rewards.narrow(0, 1, t_len);
  inputs.is_terminal = (rollout.discounts.narrow(0, 1, t_len) == 0)
                           .to(rollout.rewards.options());
  inputs.behaviour_out = rollout.agent_outs;
  inputs.agent_out = unroll(params, rollout.observations);
  const auto [meta_out, meta_state] =
      rule_->unroll_meta_net(meta_params, params, state.meta, inputs,
                             [this](const Params &p, const torch::Tensor &obs) {
                               return unroll(p, obs);
                             });

  // Differentiated through the agent's own outputs; the targets are constants
  // of this gradient, but not of the meta-gradient.
  torch::AutoGradMode loss_grad(true);
  Params own;
  for (const auto &[name, value] : params) {
    own[name] =
        meta_training ? value.view_as(value) : value.detach().requires_grad_();
  }
  const auto total =
      loss(own, inputs, rollout.discounts, meta_out, meta_training);
  const auto own_values = values(own);
  auto grads = torch::autograd::grad({total}, own_values, {},
                                     /*retain_graph=*/meta_training,
                                     /*create_graph=*/meta_training,
                                     /*allow_unused=*/true);
  for (size_t i = 0; i < grads.size(); ++i) {
    if (!grads[i].defined()) {
      grads[i] = torch::zeros_like(own_values[i]);
    }
  }

  torch::AutoGradMode update_grad(meta_training);
  auto [next_params, opt] =
      optimizer_.step(params, with_values(params, grads), state.opt);
  Tensors logs{{"total_loss", total.detach()}};
  for (const auto &[name, value] : meta_out) {
    logs["meta_out/" + name] = value;
  }
  return {{next_params, opt, meta_state}, logs};
}

ValueFunction::ValueFunction(ValueFnConfig config, int64_t observation_size)
    : config_(std::move(config)),
      net_({config_.dense, config_.head_w_init_std, {}, 0},
           {{{"value", 1}}, {}}, 0, "value_fn"),
      optimizer_{config_.learning_rate, config_.max_abs_update},
      observation_size_(observation_size) {}

ValueState
ValueFunction::initial_state(const torch::TensorOptions &options) const {
  auto params = net_.init(observation_size_, options);
  return {params, optimizer_.init(params), Ema::zeros(options),
          Ema::zeros(options)};
}

std::pair<ValueOuts, ValueState>
ValueFunction::value_outs(const ValueState &state, const Rollout &rollout,
                          const torch::Tensor &logits) const {
  ValueOptions options;
  options.discount = config_.discount;
  options.lambda = config_.td_lambda;
  options.nonlinear_transform = true;
  ValueState next = state;
  auto outs =
      get_value_outs(net_(state.params, rollout.observations).at("value"), {},
                     {}, rollout.rewards, rollout.discounts, rollout.actions,
                     logits, rollout.agent_outs.at("logits"), options,
                     MovingAverage{config_.ema_decay, config_.ema_eps},
                     next.adv_ema, next.td_ema);
  return {outs, next};
}

ValueState ValueFunction::update(const ValueState &state,
                                 const Rollout &rollout,
                                 const torch::Tensor &logits) const {
  torch::AutoGradMode enable(true);
  ValueState current = state;
  current.params = detached(state.params, /*requires_grad=*/true);
  auto [outs, next] = value_outs(current, rollout, logits.detach());
  // The raw output regresses on the normalized TD error.
  const auto net_out = net_(current.params, rollout.observations).at("value");
  const auto loss =
      (config_.outer_value_cost *
       value_loss_from_td(drop_last(net_out), outs.normalized_td.detach(), {}))
          .mean();
  const auto grads = torch::autograd::grad({loss}, values(current.params));
  torch::NoGradGuard no_grad;
  auto [params, opt] = optimizer_.step(
      current.params, with_values(current.params, grads), state.opt);
  return {detached(params), opt, detached(next.adv_ema), detached(next.td_ema)};
}

} // namespace discorl
