#include "discorl/update_rules.hpp"

#include "discorl/nn.hpp"

namespace discorl {
namespace {

torch::Tensor drop_last(const torch::Tensor &x) {
  return x.narrow(0, 0, x.size(0) - 1);
}

torch::TensorOptions options_of(const Params &params) {
  return params.begin()->second.options().requires_grad(false);
}

} // namespace

Params UpdateRule::init_params(const torch::TensorOptions &) const {
  return {};
}

torch::Tensor UpdateRule::agent_loss_no_meta(const RuleInputs &inputs,
                                             const Tensors &) const {
  return torch::zeros_like(inputs.rewards);
}

DiscoRule::DiscoRule(DiscoConfig config)
    : config_(std::move(config)), net_(config_.net) {}

ValueOptions DiscoRule::value_options() const {
  ValueOptions options;
  options.discount = config_.value_discount;
  options.lambda = config_.td_lambda;
  options.nonlinear_transform = true;
  options.categorical_value = true;
  options.max_abs_value = config_.max_abs_value;
  options.drop_last = false;
  return options;
}

OutputSpec DiscoRule::output_spec(int64_t num_actions) const {
  const int64_t p = config_.net.prediction_size;
  return {{{"logits", num_actions}, {"y", p}},
          {{"z", p}, {"aux_pi", num_actions}, {"q", config_.num_bins}}};
}

Params DiscoRule::init_params(const torch::TensorOptions &options) const {
  return net_.init(options);
}

MetaState DiscoRule::init_meta_state(const Params &agent_params) const {
  const auto options = options_of(agent_params);
  return {net_.initial_state(options), Ema::zeros(options), Ema::zeros(options),
          agent_params};
}

std::pair<Tensors, MetaState>
DiscoRule::unroll_meta_net(const Params &meta_params,
                           const Params &agent_params, const MetaState &state,
                           const RuleInputs &inputs,
                           const UnrollFn &unroll) const {
  const auto target_out = unroll(state.target_params, inputs.observations);
  const auto &out = inputs.agent_out;

  // Retrace values of the online and target Q-functions.
  MetaState next = state;
  const auto values =
      get_value_outs({}, out.at("q"), target_out.at("q"), inputs.rewards,
                     1.0 - inputs.is_terminal, inputs.actions, out.at("logits"),
                     inputs.behaviour_out.at("logits"), value_options(),
                     MovingAverage{config_.ema_decay, config_.ema_eps},
                     next.adv_ema, next.td_ema);

  MetaNetInputs in;
  in.actions = drop_last(inputs.actions);
  in.logits = out.at("logits");
  in.behaviour_logits = inputs.behaviour_out.at("logits");
  in.target_logits = target_out.at("logits");
  in.y = out.at("y");
  in.target_y = target_out.at("y");
  in.z = out.at("z");
  in.target_z = target_out.at("z");
  in.rewards = inputs.rewards;
  in.discounts = 1.0 - inputs.is_terminal;
  in.v = values.value;
  in.adv = values.adv;
  in.normalized_adv = values.normalized_adv;
  in.q = values.target_q_value;
  in.qv_adv = values.qv_adv;
  in.normalized_qv_adv = values.normalized_qv_adv;
  const auto [targets, rnn] = net_(meta_params, in, state.rnn);
  next.rnn = rnn;

  const double coeff = config_.target_params_coeff;
  for (auto &[name, target] : next.target_params) {
    target = target * coeff + (1.0 - coeff) * agent_params.at(name);
  }

  Tensors meta_out{
      {"pi", targets.pi},
      {"y", targets.y},
      {"z", targets.z},
      {"q_target", values.q_target},
      {"q_td", values.q_td},
      {"adv", values.adv},
      {"target_out/logits", target_out.at("logits")},
  };
  return {meta_out, next};
}

torch::Tensor DiscoRule::agent_loss(const RuleInputs &inputs,
                                    const Tensors &meta_out,
                                    bool backprop) const {
  const auto &out = inputs.agent_out;
  const auto actions = drop_last(inputs.actions);
  const auto target = [&](const char *name) {
    return backprop ? meta_out.at(name) : meta_out.at(name).detach();
  };
  const auto pi_loss = nn::kl(target("pi"), drop_last(out.at("logits")));
  const auto y_loss = nn::kl(target("y"), drop_last(out.at("y")));
  const auto z_loss =
      nn::kl(target("z"), nn::take(drop_last(out.at("z")), actions));
  // The auxiliary one-step prediction of the next policy.
  const auto next_logits = out.at("logits").narrow(0, 1, actions.size(0));
  const auto aux_loss = nn::kl(next_logits.detach(),
                               nn::take(drop_last(out.at("aux_pi")), actions)) *
                        (1.0 - inputs.is_terminal);
  return config_.pi_cost * pi_loss + config_.y_cost * y_loss +
         config_.z_cost * z_loss + config_.aux_policy_cost * aux_loss;
}

torch::Tensor DiscoRule::agent_loss_no_meta(const RuleInputs &inputs,
                                            const Tensors &meta_out) const {
  const auto q_taken =
      drop_last(nn::take(inputs.agent_out.at("q"), inputs.actions));
  return config_.value_cost * value_loss_from_td(q_taken,
                                                 meta_out.at("q_td").detach(),
                                                 value_options());
}

ActorCritic::ActorCritic(ActorCriticConfig config)
    : config_(std::move(config)) {}

ValueOptions ActorCritic::value_options() const {
  ValueOptions options;
  options.discount = config_.discount;
  options.lambda = config_.vtrace_lambda;
  options.nonlinear_transform = config_.nonlinear_value_transform;
  options.categorical_value = config_.categorical_value;
  options.max_abs_value = config_.max_abs_value;
  options.drop_last = false;
  return options;
}

OutputSpec ActorCritic::output_spec(int64_t num_actions) const {
  return {{{"logits", num_actions},
           {"v", config_.categorical_value ? config_.num_bins : 1}},
          {}};
}

MetaState ActorCritic::init_meta_state(const Params &agent_params) const {
  const auto options = options_of(agent_params);
  return {{}, Ema::zeros(options), Ema::zeros(options), {}};
}

std::pair<Tensors, MetaState>
ActorCritic::unroll_meta_net(const Params &, const Params &,
                             const MetaState &state, const RuleInputs &inputs,
                             const UnrollFn &) const {
  MetaState next = state;
  const auto values = get_value_outs(
      inputs.agent_out.at("v"), {}, {}, inputs.rewards,
      1.0 - inputs.is_terminal, inputs.actions, inputs.agent_out.at("logits"),
      inputs.behaviour_out.at("logits"), value_options(),
      MovingAverage{config_.ema_decay, config_.ema_eps}, next.adv_ema,
      next.td_ema);
  Tensors meta_out{
      {"adv", config_.normalize_adv ? values.normalized_adv : values.adv},
      {"td", config_.normalize_td ? values.normalized_td : values.td},
  };
  return {meta_out, next};
}

torch::Tensor ActorCritic::agent_loss(const RuleInputs &inputs,
                                      const Tensors &meta_out, bool) const {
  const auto actions = drop_last(inputs.actions);
  const auto logits = drop_last(inputs.agent_out.at("logits"));
  const auto value_loss =
      value_loss_from_td(drop_last(inputs.agent_out.at("v")),
                         meta_out.at("td").detach(), value_options());
  const auto pg_loss =
      -nn::log_prob(logits, actions) * meta_out.at("adv").detach();
  return config_.pg_cost * pg_loss + config_.value_cost * value_loss -
         config_.entropy_cost * nn::entropy(logits);
}

} // namespace discorl
