#include "discorl/meta_train.hpp"

#include "discorl/nn.hpp"

namespace discorl {
namespace {

torch::Tensor drop_last(const torch::Tensor &x) {
  return x.narrow(0, 0, x.size(0) - 1);
}

Ema detached(const Ema &ema) {
  return {ema.moment1.detach(), ema.moment2.detach(),
          ema.decay_product.detach()};
}

LearnerState detached(const LearnerState &state) {
  LearnerState out{discorl::detached(state.params),
                   {state.opt.count, discorl::detached(state.opt.mu),
                    discorl::detached(state.opt.nu)},
                   state.meta};
  if (out.meta.rnn.h.defined()) {
    out.meta.rnn = {out.meta.rnn.h.detach(), out.meta.rnn.c.detach()};
  }
  out.meta.adv_ema = detached(out.meta.adv_ema);
  out.meta.td_ema = detached(out.meta.td_ema);
  out.meta.target_params = discorl::detached(out.meta.target_params);
  return out;
}

// Mean over [N, T, B, ...] per inner step, squared, averaged over steps.
torch::Tensor squared_mean(const std::vector<torch::Tensor> &targets) {
  return torch::stack(targets).flatten(1).mean(1).square().mean();
}

} // namespace

MetaGradient meta_gradient(const Agent &agent, const ValueFunction &value_fn,
                           const Params &meta_params,
                           const LearnerState &learner, const ValueState &value,
                           const std::vector<Rollout> &train,
                           const Rollout &valid,
                           const MetaTrainConfig &config) {
  torch::AutoGradMode enable(true);
  LearnerState state = learner;
  ValueState value_state = value;
  std::vector<torch::Tensor> pi, y, z, kl;
  for (const auto &rollout : train) {
    // The value function learns the pre-update policy's values.
    const auto logits =
        agent.unroll(state.params, rollout.observations).at("logits").detach();
    auto [next, logs] =
        agent.learner_step(rollout, state, meta_params, /*meta_training=*/true);
    value_state = value_fn.update(value_state, rollout, logits);
    pi.push_back(logs.at("meta_out/pi"));
    y.push_back(logs.at("meta_out/y"));
    z.push_back(logs.at("meta_out/z"));
    kl.push_back(
        nn::kl(drop_last(logs.at("meta_out/target_out/logits")).detach(),
               logs.at("meta_out/pi"))
            .mean());
    state = std::move(next);
  }

  // The updated agent's policy gradient on the validation rollout.
  const auto out = agent.unroll(state.params, valid.observations);
  const auto logits = drop_last(out.at("logits"));
  const auto actions = drop_last(valid.actions);
  const auto adv =
      value_fn.value_outs(value_state, valid, out.at("logits").detach())
          .first.normalized_adv.detach();
  const auto pg_loss = (-nn::log_prob(logits, actions) * adv).mean();
  const auto entropy = nn::entropy(logits).mean();
  auto reg_loss = -config.entropy_cost * entropy;
  reg_loss =
      reg_loss +
      config.prediction_entropy_cost *
          (-nn::entropy(out.at("y")).mean() -
           nn::entropy(nn::take(drop_last(out.at("z")), actions)).mean());
  reg_loss =
      reg_loss + config.target_mean_cost *
                     (squared_mean(y) + squared_mean(z) + squared_mean(pi));
  reg_loss = reg_loss + config.target_kl_cost * torch::stack(kl).mean();
  const auto loss = pg_loss + reg_loss;

  const auto meta_values = values(meta_params);
  auto grads = torch::autograd::grad({loss}, meta_values, {},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false,
                                     /*allow_unused=*/true);
  for (size_t i = 0; i < grads.size(); ++i) {
    if (!grads[i].defined()) {
      grads[i] = torch::zeros_like(meta_values[i]);
    }
  }
  Tensors logs{{"meta_loss", loss.detach()},
               {"pg_loss", pg_loss.detach()},
               {"reg_loss", reg_loss.detach()},
               {"entropy", entropy.detach()}};
  return {loss.detach(), with_values(meta_params, grads), detached(state),
          value_state, logs};
}

MetaTrainer::MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
                         ValueFnConfig value_config, DiscoConfig rule_config,
                         const EnvironmentFactory &make_env, Params meta_params,
                         const torch::TensorOptions &options, uint64_t seed)
    : config_(std::move(config)),
      meta_params_(discorl::detached(to(meta_params, options), true)) {
  const auto probe = make_env(1, seed);
  agent_ = std::make_unique<Agent>(
      std::move(agent_config),
      std::make_shared<DiscoRule>(std::move(rule_config)),
      probe->observation_size(), probe->num_actions());
  value_fn_ = std::make_unique<ValueFunction>(std::move(value_config),
                                              probe->observation_size());
  optimizer_ = std::make_unique<torch::optim::Adam>(
      values(meta_params_), torch::optim::AdamOptions(config_.learning_rate));
  for (int64_t i = 0; i < config_.num_agents; ++i) {
    Member member;
    member.env = make_env(config_.batch_size, seed + 1 + i);
    member.timestep = member.env->reset();
    member.learner = agent_->initial_learner_state(options);
    member.value = value_fn_->initial_state(options);
    members_.push_back(std::move(member));
  }
}

Tensors MetaTrainer::step() {
  // Every agent acts with its current parameters, then learns.
  std::vector<torch::Tensor> grads;
  Tensors logs;
  double rewards = 0.0, positive = 0.0, negative = 0.0;
  for (auto &member : members_) {
    std::vector<Rollout> train;
    for (int64_t i = 0; i < config_.num_inner_steps; ++i) {
      train.push_back(collect(*agent_, member.learner.params, *member.env,
                              member.timestep, config_.rollout_len));
      rewards += train.back().rewards.mean().item<double>();
      positive += (train.back().rewards > 0).sum().item<double>();
      negative += (train.back().rewards < 0).sum().item<double>();
    }
    const auto valid = collect(*agent_, member.learner.params, *member.env,
                               member.timestep, 2 * config_.rollout_len);
    auto result =
        meta_gradient(*agent_, *value_fn_, meta_params_, member.learner,
                      member.value, train, valid, config_);
    member.learner = std::move(result.learner);
    member.value = std::move(result.value);
    const auto member_grads = values(result.grads);
    if (grads.empty()) {
      grads = member_grads;
    } else {
      for (size_t i = 0; i < grads.size(); ++i) {
        grads[i] = grads[i] + member_grads[i];
      }
    }
    for (const auto &[name, value] : result.logs) {
      logs[name] = logs.count(name) ? logs[name] + value : value;
    }
  }

  const double n = static_cast<double>(members_.size());
  auto params = values(meta_params_);
  torch::Tensor squared_norm = torch::zeros({}, grads.front().options());
  for (size_t i = 0; i < params.size(); ++i) {
    params[i].mutable_grad() = grads[i] / n;
    squared_norm = squared_norm + (grads[i] / n).square().sum();
  }
  optimizer_->step();
  optimizer_->zero_grad();

  for (auto &[name, value] : logs) {
    value = value / n;
  }
  logs["meta_grad_norm"] = squared_norm.sqrt();
  logs["rewards"] = torch::tensor(rewards / (n * config_.num_inner_steps));
  logs["pos_rewards"] = torch::tensor(positive / n);
  logs["neg_rewards"] = torch::tensor(negative / n);
  return logs;
}

} // namespace discorl
