#include "discorl/networks.hpp"

#include <algorithm>

#include "discorl/nn.hpp"
#include "discorl/value.hpp"

namespace discorl {
namespace {

// Haiku numbers repeated module names: name, name_1, name_2, ...
std::string numbered(const std::string &name, size_t i) {
  return i == 0 ? name : name + "_" + std::to_string(i);
}

std::vector<int64_t> append(std::vector<int64_t> sizes, int64_t size) {
  sizes.push_back(size);
  return sizes;
}

// meta_nets._construct_input with Disco103's input option: per-step inputs
// [T, B, ...] and the action-conditional embedding [T, B, A, C]. Its
// stop_grad transforms are the detach() calls; y and z keep gradients.
std::pair<torch::Tensor, torch::Tensor>
construct_input(const Scope &s, const MetaNetInputs &in,
                const std::vector<int64_t> &prediction_embedding,
                const std::vector<int64_t> &policy_channels) {
  const int64_t t = in.rewards.size(0);
  const int64_t num_actions = in.logits.size(-1);
  const auto &a = in.actions;
  const auto y_net = [&](const torch::Tensor &x) {
    return nn::mlp(s / "mlp", x.softmax(-1), prediction_embedding);
  };
  const auto z_net = [&](const torch::Tensor &x) {
    return nn::mlp(s / "mlp_1", x.softmax(-1), prediction_embedding);
  };
  const auto now = [t](const torch::Tensor &x) { return x.narrow(0, 0, t); };
  const auto pair = [t](const torch::Tensor &x) {
    return torch::cat({x.narrow(0, 0, t), x.narrow(0, 1, t)}, -1);
  };
  const auto taken = [&a](const torch::Tensor &x) { return nn::take(x, a); };
  const auto scalar = [](const torch::Tensor &x) { return x.unsqueeze(-1); };
  const auto policy = in.logits.softmax(-1).detach();
  const auto pi_avg = [&policy](const torch::Tensor &x) {
    return (x * policy.unsqueeze(-1)).sum(2);
  };
  const auto max_a = [](const torch::Tensor &x) {
    return std::get<0>(x.max(2));
  };

  const auto pi = policy;
  const auto mu = in.behaviour_logits.softmax(-1).detach();
  const auto target_pi = in.target_logits.softmax(-1).detach();
  const auto z = z_net(in.z); // [T+1, B, A, E]
  const auto target_z = z_net(in.target_z);
  std::vector<torch::Tensor> inputs{
      scalar(taken(now(pi))),
      scalar(taken(now(mu))),
      scalar(signed_logp1(in.rewards)),
      scalar(in.discounts),
      pair(scalar(signed_logp1(in.v))).detach(),
      scalar(signed_logp1(in.adv)).detach(),
      scalar(in.normalized_adv).detach(),
      scalar(taken(now(target_pi))),
      pair(y_net(in.y)),
      pair(y_net(in.target_y)),
      taken(now(z)),
      pair(pi_avg(z)),
      pair(max_a(z)),
      taken(now(target_z)),
      pair(pi_avg(target_z)),
      pair(max_a(target_z)),
  };
  const auto per_action = torch::cat(
      {scalar(now(pi)), scalar(now(mu)), scalar(now(target_pi)), now(z),
       now(target_z), scalar(now(signed_logp1(in.q))).detach(),
       scalar(now(signed_logp1(in.qv_adv))).detach(),
       scalar(now(in.normalized_qv_adv)).detach(),
       scalar(torch::one_hot(a.to(torch::kLong), num_actions)
                  .to(in.rewards.options()))},
      -1);
  const auto embedding =
      nn::action_conv(s / "sequential", per_action, policy_channels);
  inputs.push_back(embedding.mean(2));
  inputs.push_back(taken(embedding));
  return {torch::cat(inputs, -1), embedding};
}

} // namespace

AgentNet::AgentNet(NetConfig config, OutputSpec spec, int64_t num_actions,
                   std::string name)
    : config_(std::move(config)), spec_(std::move(spec)),
      num_actions_(num_actions), name_(std::move(name)) {
  std::sort(spec_.flat.begin(), spec_.flat.end());
}

Params AgentNet::init(int64_t observation_size,
                      const torch::TensorOptions &options) const {
  Params params;
  apply(Scope(params, options), torch::zeros({1, observation_size}, options),
        true);
  return params;
}

Tensors AgentNet::operator()(const Params &params,
                             const torch::Tensor &observations,
                             bool model) const {
  return apply(Scope(params), observations, model);
}

Tensors AgentNet::apply(const Scope &root, const torch::Tensor &observations,
                        bool model) const {
  const Scope s = root / name_;
  auto lead = observations.sizes().vec();
  lead.pop_back();
  const auto x = observations.reshape({-1, observations.size(-1)});
  const auto torso = nn::mlp(s / "~_embedding_pass/torso", x, config_.dense);

  Tensors out;
  const auto with_lead = [&lead](const torch::Tensor &y, int64_t trailing) {
    auto shape = lead;
    for (int64_t i = y.dim() - trailing; i < y.dim(); ++i) {
      shape.push_back(y.size(i));
    }
    return y.reshape(shape);
  };
  for (size_t i = 0; i < spec_.flat.size(); ++i) {
    const auto &[name, size] = spec_.flat[i];
    out[name] =
        with_lead(nn::mlp(s / ("~_head_pass/" + numbered("torso_head", i)),
                          torso, {size}, config_.head_w_init_std),
                  1);
  }
  if (!model || spec_.model.empty()) {
    return out;
  }
  // One LSTM step from the state for every action.
  const int64_t n = x.size(0);
  const auto cell = nn::linear(s / "linear", torso, config_.lstm_size);
  const auto one_hot = torch::eye(num_actions_, x.options()).repeat({n, 1});
  const auto [h, c] =
      nn::lstm(s / "action_cond", one_hot,
               torch::tanh(cell).repeat_interleave(num_actions_, 0),
               cell.repeat_interleave(num_actions_, 0));
  for (size_t i = 0; i < spec_.model.size(); ++i) {
    const auto &[name, size] = spec_.model[i];
    const auto y = nn::mlp(s / numbered("mlp", i), h,
                           append(config_.model_head_hiddens, size));
    out[name] = with_lead(y.view({n, num_actions_, size}), 2);
  }
  return out;
}

MetaNet::MetaNet(MetaNetConfig config) : config_(std::move(config)) {}

Params MetaNet::init(const torch::TensorOptions &options) const {
  constexpr int64_t t = 2;
  constexpr int64_t a = 3; // parameters do not depend on the action count
  const auto zeros = [&](at::IntArrayRef shape) {
    return torch::zeros(shape, options);
  };
  const int64_t p = config_.prediction_size;
  MetaNetInputs in;
  in.actions = torch::zeros({t, 1}, options.dtype(torch::kLong));
  in.logits = in.behaviour_logits = in.target_logits = zeros({t + 1, 1, a});
  in.y = in.target_y = zeros({t + 1, 1, p});
  in.z = in.target_z = zeros({t + 1, 1, a, p});
  in.rewards = in.adv = in.normalized_adv = zeros({t, 1});
  in.discounts = torch::ones({t, 1}, options);
  in.v = zeros({t + 1, 1});
  in.q = in.qv_adv = in.normalized_qv_adv = zeros({t + 1, 1, a});
  Params params;
  apply(Scope(params, options), in, initial_state(options));
  return params;
}

LstmState MetaNet::initial_state(const torch::TensorOptions &options) const {
  return {torch::zeros({config_.meta_hidden_size}, options),
          torch::zeros({config_.meta_hidden_size}, options)};
}

std::pair<MetaNetOutput, LstmState>
MetaNet::operator()(const Params &params, const MetaNetInputs &in,
                    const LstmState &state) const {
  return apply(Scope(params), in, state);
}

std::pair<MetaNetOutput, LstmState>
MetaNet::apply(const Scope &root, const MetaNetInputs &in,
               const LstmState &state) const {
  const Scope s = root / "lstm";
  const auto [x, embedding] =
      construct_input(s, in, config_.embedding_size, config_.policy_channels);

  // The per-trajectory LSTM runs backwards, reset where an episode ends.
  const int64_t t_len = in.rewards.size(0);
  auto h = torch::zeros({x.size(1), config_.hidden_size}, x.options());
  auto c = torch::zeros_like(h);
  std::vector<torch::Tensor> hidden(t_len);
  for (int64_t t = t_len - 1; t >= 0; --t) {
    const auto keep = in.discounts[t].unsqueeze(-1);
    std::tie(h, c) = nn::lstm(s / "lstm", x[t], h * keep, c * keep);
    hidden[t] = h;
  }
  // Multiplicative interaction with the lifetime meta-RNN.
  const auto gated = torch::stack(hidden) * nn::linear(s / "linear", state.h,
                                                       config_.hidden_size,
                                                       config_.state_stddev);

  MetaNetOutput out;
  out.meta_input_emb =
      nn::linear(s / "linear_1", gated, 1, config_.output_stddev);
  out.y = nn::linear(s / "linear_2", gated, config_.prediction_size,
                     config_.aux_stddev);
  out.z = nn::linear(s / "linear_3", gated, config_.prediction_size,
                     config_.aux_stddev);
  const auto per_action = torch::cat(
      {gated.unsqueeze(2).expand({-1, -1, embedding.size(2), -1}), embedding},
      -1);
  out.pi = nn::linear(s / "linear_4",
                      nn::action_conv(s / "sequential_1", per_action,
                                      config_.policy_target_channels),
                      1, config_.policy_target_stddev)
               .squeeze(-1);

  // The lifetime meta-RNN steps once on the batch-time average.
  const Scope m = s / "~/meta_lstm/~unroll";
  const auto meta_x = construct_input(m, in, config_.meta_pred_embedding_size,
                                      config_.meta_policy_channels)
                          .first;
  const auto y_emb =
      nn::mlp(m / "mlp", out.y.softmax(-1), config_.meta_pred_embedding_size);
  const auto summary =
      nn::mlp(m / "mlp_2", torch::cat({meta_x, out.meta_input_emb, y_emb}, -1),
              config_.meta_embedding_size)
          .flatten(0, 1)
          .mean(0);
  const auto [next_h, next_c] = nn::lstm(m / "lstm", summary, state.h, state.c);
  return {out, {next_h, next_c}};
}

} // namespace discorl
