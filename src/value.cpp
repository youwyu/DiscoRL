#include "discorl/value.hpp"

#include <vector>

#include "discorl/nn.hpp"

namespace discorl {
namespace {

constexpr double kHyperbolicEps = 1e-3;

torch::Tensor drop_last(const torch::Tensor &x) {
  return x.narrow(0, 0, x.size(0) - 1);
}

torch::Tensor drop_first(const torch::Tensor &x) {
  return x.narrow(0, 1, x.size(0) - 1);
}

// V-trace (rlax.vtrace_td_error_and_advantage) over values [T+1, B].
void estimate_values(const torch::Tensor &rewards,
                     const torch::Tensor &discounts, const torch::Tensor &rho,
                     const torch::Tensor &values,
                     const torch::Tensor &target_values, double lambda,
                     ValueOuts &out) {
  const auto v_tm1 = drop_last(target_values);
  const auto v_t = drop_first(target_values);
  const auto clipped_rho = rho.clamp_max(1.0);
  const auto trace = lambda * clipped_rho;
  const auto td_errors = clipped_rho * (rewards + discounts * v_t - v_tm1);
  const int64_t t_len = rewards.size(0);
  std::vector<torch::Tensor> errors(t_len);
  auto acc = torch::zeros_like(v_tm1[0]);
  for (int64_t t = t_len - 1; t >= 0; --t) {
    acc = td_errors[t] + discounts[t] * trace[t] * acc;
    errors[t] = acc;
  }
  // Stop gradients through the bootstrap targets.
  const auto targets_tm1 = (torch::stack(errors) + v_tm1).detach();
  const auto vtrace_errors = targets_tm1 - v_tm1;
  const auto bootstrap = torch::cat(
      {lambda * drop_first(targets_tm1) + (1.0 - lambda) * drop_first(v_tm1),
       v_t.narrow(0, t_len - 1, 1)});
  const auto q_estimate = rewards + discounts * bootstrap;
  out.adv = clipped_rho * (q_estimate - v_tm1);
  out.value_target = vtrace_errors + v_tm1;
  out.td = out.value_target - drop_last(values);
}

// Retrace (rlax.general_off_policy_returns_from_q_and_v as disco_rl calls
// it): G_t = r_t + d_t (V_{t+1} + c_{t+1} (G_{t+1} - Q_{t+1})), G = r + d V_T.
torch::Tensor retrace(const torch::Tensor &rewards,
                      const torch::Tensor &discounts,
                      const torch::Tensor &trace, const torch::Tensor &v,
                      const torch::Tensor &q_taken) {
  const int64_t t_len = rewards.size(0);
  std::vector<torch::Tensor> returns(t_len);
  auto g = rewards[t_len - 1] + discounts[t_len - 1] * v[t_len];
  returns[t_len - 1] = g;
  for (int64_t t = t_len - 2; t >= 0; --t) {
    g = rewards[t] +
        discounts[t] * (v[t + 1] + trace[t + 1] * (g - q_taken[t + 1]));
    returns[t] = g;
  }
  return torch::stack(returns);
}

} // namespace

torch::Tensor signed_logp1(const torch::Tensor &x) {
  return torch::sign(x) * torch::log1p(torch::abs(x));
}

torch::Tensor signed_hyperbolic(const torch::Tensor &x) {
  return torch::sign(x) * (torch::sqrt(torch::abs(x) + 1.0) - 1.0) +
         kHyperbolicEps * x;
}

torch::Tensor signed_parabolic(const torch::Tensor &x) {
  constexpr double eps = kHyperbolicEps;
  const auto z =
      torch::sqrt(1.0 + 4.0 * eps * (eps + 1.0 + torch::abs(x))) / 2.0 / eps -
      1.0 / 2.0 / eps;
  return torch::sign(x) * (z.square() - 1.0);
}

torch::Tensor to_two_hot(const torch::Tensor &x, double min, double max,
                         int64_t num_bins) {
  const auto scalar = x.clamp(min, max);
  const auto bin = (scalar - min) / (max - min) * (num_bins - 1);
  const auto lower = torch::floor(bin);
  const auto upper = torch::ceil(bin);
  const auto lower_value = lower / (num_bins - 1.0) * (max - min) + min;
  const auto upper_value = upper / (num_bins - 1.0) * (max - min) + min;
  const auto p_lower =
      (upper_value - scalar) / (upper_value - lower_value + 1e-5);
  auto one_hot = [&](const torch::Tensor &index) {
    return torch::one_hot(index.detach().to(torch::kLong), num_bins)
        .to(x.options());
  };
  return one_hot(lower) * p_lower.unsqueeze(-1) +
         one_hot(upper) * (1.0 - p_lower).unsqueeze(-1);
}

torch::Tensor from_two_hot(const torch::Tensor &probs, double min, double max) {
  const auto support =
      torch::linspace(min, max, probs.size(-1), probs.options());
  return (probs * support).sum(-1);
}

Ema Ema::zeros(const torch::TensorOptions &options) {
  return {torch::zeros({}, options), torch::zeros({}, options),
          torch::ones({}, options)};
}

Ema MovingAverage::update(const Ema &state, const torch::Tensor &value) const {
  return {decay * state.moment1 + (1.0 - decay) * value.mean(),
          decay * state.moment2 + (1.0 - decay) * value.square().mean(),
          state.decay_product * decay};
}

torch::Tensor MovingAverage::normalize(const torch::Tensor &value,
                                       const Ema &state,
                                       bool subtract_mean) const {
  const auto debias = 1.0 / (1.0 - state.decay_product);
  const auto mean = state.moment1 * debias;
  const auto variance = (state.moment2 * debias - mean.square()).clamp_min(0.0);
  const auto scale = torch::sqrt(variance + 1e-12) + eps;
  return subtract_mean ? (value - mean) / scale : value / scale;
}

torch::Tensor scalar_values(const torch::Tensor &net_out,
                            const ValueOptions &options) {
  auto v = options.categorical_value
               ? from_two_hot(net_out.softmax(-1), -options.max_abs_value,
                              options.max_abs_value)
               : net_out.squeeze(-1);
  return options.nonlinear_transform ? signed_parabolic(v) : v;
}

ValueOuts
get_value_outs(const torch::Tensor &value_net_out,
               const torch::Tensor &q_net_out,
               const torch::Tensor &target_q_net_out,
               const torch::Tensor &rewards, const torch::Tensor &env_discounts,
               const torch::Tensor &actions, const torch::Tensor &pi_logits,
               const torch::Tensor &mu_logits, const ValueOptions &options,
               const MovingAverage &ema, Ema &adv_ema, Ema &td_ema) {
  const auto r = options.drop_last ? drop_last(rewards) : rewards;
  const auto discounts =
      (options.drop_last ? drop_last(env_discounts) : env_discounts) *
      options.discount;
  const auto taken = drop_last(actions);

  ValueOuts out;
  out.rho = (nn::log_prob(drop_last(pi_logits), taken) -
             nn::log_prob(drop_last(mu_logits), taken))
                .exp()
                .detach();
  const auto clipped_rho = out.rho.clamp_max(1.0);
  if (q_net_out.defined()) {
    out.q_value = scalar_values(q_net_out, options);
    out.target_q_value = target_q_net_out.defined()
                             ? scalar_values(target_q_net_out, options)
                             : out.q_value;
    const auto pi = pi_logits.softmax(-1);
    out.value = (pi * out.q_value).sum(-1);
    out.target_value = (pi * out.target_q_value).sum(-1);
    const auto v_tm1 = drop_last(out.target_value);
    const auto q_taken = nn::take(drop_last(out.q_value), taken);
    out.q_target =
        retrace(r, discounts, options.lambda * clipped_rho, out.target_value,
                nn::take(drop_last(out.target_q_value), taken))
            .detach();
    out.qv_adv = out.target_q_value - out.target_value.unsqueeze(-1);
    out.value_target = v_tm1 + clipped_rho * (out.q_target - v_tm1);
    out.adv = out.q_target - v_tm1;
    out.q_td = out.q_target - q_taken;
    out.td = out.value_target - drop_last(out.value);
  } else {
    out.value = scalar_values(value_net_out, options);
    out.target_value = out.value;
    estimate_values(r, discounts, out.rho, out.value, out.target_value,
                    options.lambda, out);
  }

  adv_ema = ema.update(adv_ema, out.adv);
  out.normalized_adv = ema.normalize(out.adv, adv_ema);
  if (q_net_out.defined()) {
    out.normalized_qv_adv = ema.normalize(out.qv_adv, adv_ema);
    td_ema = ema.update(td_ema, out.q_td);
    out.normalized_q_td = ema.normalize(out.q_td, td_ema, false);
    out.normalized_td = torch::zeros_like(out.td);
  } else {
    td_ema = ema.update(td_ema, out.td);
    out.normalized_td = ema.normalize(out.td, td_ema, false);
  }
  return out;
}

torch::Tensor value_loss_from_td(const torch::Tensor &net_out,
                                 const torch::Tensor &td,
                                 const ValueOptions &options) {
  auto target = (scalar_values(net_out, options) + td).detach();
  if (options.nonlinear_transform) {
    target = signed_hyperbolic(target);
  }
  if (options.categorical_value) {
    const auto probs = to_two_hot(target, -options.max_abs_value,
                                  options.max_abs_value, net_out.size(-1));
    return -(probs * net_out.log_softmax(-1)).sum(-1);
  }
  return 0.5 * (net_out.squeeze(-1) - target).square();
}

} // namespace discorl
