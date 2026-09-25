#pragma once

#include <torch/torch.h>

// Value estimation (disco_rl/value_fns/value_utils.py, utils.MovingAverage).
// Tensors are time-major, [T, B, ...].
namespace discorl {

// rlax transforms: sign(x) log(|x| + 1), and h(x) with its inverse h^-1.
torch::Tensor signed_logp1(const torch::Tensor &x);
torch::Tensor signed_hyperbolic(const torch::Tensor &x);
torch::Tensor signed_parabolic(const torch::Tensor &x);
// Two-hot categoricals over num_bins points evenly spaced in [min, max].
torch::Tensor to_two_hot(const torch::Tensor &x, double min, double max,
                         int64_t num_bins);
torch::Tensor from_two_hot(const torch::Tensor &probs, double min, double max);

// Debiased exponential moving moments of a scalar statistic.
struct Ema {
  torch::Tensor moment1, moment2, decay_product;
  static Ema zeros(const torch::TensorOptions &options);
};

struct MovingAverage {
  double decay = 0.99;
  double eps = 1e-6;
  Ema update(const Ema &state, const torch::Tensor &value) const;
  torch::Tensor normalize(const torch::Tensor &value, const Ema &state,
                          bool subtract_mean = true) const;
};

// Outputs of get_value_outs; the Q-value fields only with Q-functions.
struct ValueOuts {
  torch::Tensor value;             // [T+1, B]
  torch::Tensor target_value;      // [T+1, B]
  torch::Tensor rho;               // [T, B] importance weights
  torch::Tensor adv;               // [T, B]
  torch::Tensor normalized_adv;    // [T, B]
  torch::Tensor value_target;      // [T, B]
  torch::Tensor td;                // [T, B]
  torch::Tensor normalized_td;     // [T, B]
  torch::Tensor q_value;           // [T+1, B, A]
  torch::Tensor target_q_value;    // [T+1, B, A]
  torch::Tensor qv_adv;            // [T+1, B, A]
  torch::Tensor normalized_qv_adv; // [T+1, B, A]
  torch::Tensor q_target;          // [T, B] Retrace
  torch::Tensor q_td;              // [T, B]
  torch::Tensor normalized_q_td;   // [T, B]
};

struct ValueOptions {
  double discount = 0.997;
  double lambda = 0.95;
  bool nonlinear_transform = false; // values are h(v)
  bool categorical_value = false;   // two-hot logits over +-max_abs_value
  double max_abs_value = 300.0;
  // Rewards and discounts cover all T+1 steps, of which the last is dropped.
  bool drop_last = true;
};

// V-trace from state values [T+1, B, 1 or bins], or Retrace from Q-values
// [T+1, B, A, 1 or bins] (and their targets, which may be undefined).
// actions and pi/mu logits span T+1 steps; the EMAs normalize advantages and
// TD errors and are advanced in place.
ValueOuts
get_value_outs(const torch::Tensor &value_net_out,
               const torch::Tensor &q_net_out,
               const torch::Tensor &target_q_net_out,
               const torch::Tensor &rewards, const torch::Tensor &env_discounts,
               const torch::Tensor &actions, const torch::Tensor &pi_logits,
               const torch::Tensor &mu_logits, const ValueOptions &options,
               const MovingAverage &ema, Ema &adv_ema, Ema &td_ema);

// Scalar values of network outputs [T, B, 1 or bins].
torch::Tensor scalar_values(const torch::Tensor &net_out,
                            const ValueOptions &options);
// Per-step loss towards stop_gradient(value + td).
torch::Tensor value_loss_from_td(const torch::Tensor &net_out,
                                 const torch::Tensor &td,
                                 const ValueOptions &options);

} // namespace discorl
