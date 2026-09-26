#include "discorl/meta_train.hpp"

#include <exception>
#include <limits>
#include <numeric>
#include <thread>

#include <c10/core/DeviceGuard.h>

#if __has_include(<c10/cuda/CUDAGuard.h>)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#define DISCORL_CUDA_STREAMS 1
#endif

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

// Every tensor of a learner state, in a fixed order.
std::vector<torch::Tensor *> state_tensors(LearnerState &state) {
  std::vector<torch::Tensor *> out;
  auto add = [&out](Params &params) {
    for (auto &[name, value] : params) {
      out.push_back(&value);
    }
  };
  add(state.params);
  add(state.opt.mu);
  add(state.opt.nu);
  if (state.meta.rnn.h.defined()) {
    out.push_back(&state.meta.rnn.h);
    out.push_back(&state.meta.rnn.c);
  }
  for (Ema *ema : {&state.meta.adv_ema, &state.meta.td_ema}) {
    out.push_back(&ema->moment1);
    out.push_back(&ema->moment2);
    out.push_back(&ema->decay_product);
  }
  add(state.meta.target_params);
  return out;
}

std::vector<torch::Tensor> tensors_of(LearnerState &state) {
  std::vector<torch::Tensor> out;
  for (auto *tensor : state_tensors(state)) {
    out.push_back(*tensor);
  }
  return out;
}

void move(LearnerState &state, const torch::Device &device) {
  for (auto *tensor : state_tensors(state)) {
    *tensor = tensor->to(device);
  }
}

void move(AdamState &state, const torch::Device &device) {
  for (Params *params : {&state.mu, &state.nu}) {
    for (auto &[name, value] : *params) {
      value = value.to(device);
    }
  }
}

void move(ValueState &state, const torch::Device &device) {
  for (Params *params : {&state.params, &state.opt.mu, &state.opt.nu}) {
    for (auto &[name, value] : *params) {
      value = value.to(device);
    }
  }
  for (Ema *ema : {&state.adv_ema, &state.td_ema}) {
    for (torch::Tensor *t : {&ema->moment1, &ema->moment2, &ema->decay_product}) {
      *t = t->to(device);
    }
  }
}

// A copy whose tensors are leaves requiring gradients.
LearnerState leaves(const LearnerState &state) {
  LearnerState out = state;
  for (auto *tensor : state_tensors(out)) {
    *tensor = tensor->detach().requires_grad_(tensor->is_floating_point());
  }
  return out;
}

// One update's share of the meta-objective's regularizers on the rule's
// targets: their squared means and KL(target policy || pi target).
torch::Tensor update_regularizer(const Tensors &logs,
                                 const MetaTrainConfig &config) {
  const auto &pi = logs.at("meta_out/pi");
  const auto kl =
      nn::kl(drop_last(logs.at("meta_out/target_out/logits")).detach(), pi)
          .mean();
  const auto means = logs.at("meta_out/y").mean().square() +
                     logs.at("meta_out/z").mean().square() +
                     pi.mean().square();
  return (config.target_mean_cost * means + config.target_kl_cost * kl) /
         static_cast<double>(config.num_inner_steps);
}

// The updated agent's advantage actor-critic loss, with its entropy terms,
// on a rollout it acted; and the meta-value function after learning from it.
struct Objective {
  torch::Tensor loss, pg_loss, entropy;
  ValueState value;
};

Objective objective(const Agent &agent, const ValueFunction &value_fn,
                    const ValueState &value, const Params &params,
                    const Rollout &rollout, const MetaTrainConfig &config) {
  const auto out = agent.unroll(params, rollout.observations);
  const auto logits = drop_last(out.at("logits"));
  const auto actions = drop_last(rollout.actions);
  const auto behaviour = out.at("logits").detach();
  const auto adv =
      value_fn.value_outs(value, rollout, behaviour).first.normalized_adv.detach();
  Objective result;
  result.pg_loss = (-nn::log_prob(logits, actions) * adv).mean();
  result.entropy = nn::entropy(logits).mean();
  result.loss =
      result.pg_loss - config.entropy_cost * result.entropy +
      config.prediction_entropy_cost *
          (-nn::entropy(out.at("y")).mean() -
           nn::entropy(nn::take(drop_last(out.at("z")), actions)).mean());
  result.value = value_fn.update(value, rollout, behaviour);
  return result;
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

MetaGradient discovery_gradient(const Agent &agent,
                                const ValueFunction &value_fn,
                                const Params &meta_params,
                                const LearnerState &learner,
                                const ValueState &value,
                                const BatchFn &next_batch, const ActFn &act,
                                const MetaTrainConfig &config) {
  torch::AutoGradMode enable(true);
  const int64_t updates = config.num_inner_steps;
  const auto meta_values = values(meta_params);
  std::vector<torch::Tensor> grads(meta_values.size());
  auto accumulate = [&grads](const std::vector<torch::Tensor> &g,
                             size_t offset) {
    for (size_t i = 0; i < grads.size(); ++i) {
      if (g[offset + i].defined()) {
        grads[i] = grads[i].defined() ? grads[i] + g[offset + i]
                                      : g[offset + i];
      }
    }
  };
  torch::Tensor reg_loss = torch::zeros({}, meta_values.front().options());
  LearnerState state = learner;
  Objective final;

  if (!config.recompute) {
    // One graph through every update.
    for (int64_t k = 0; k < updates; ++k) {
      const auto batch = next_batch(k, discorl::detached(state.params));
      auto [next, logs] = agent.learner_step(batch, state, meta_params, true);
      reg_loss = reg_loss + update_regularizer(logs, config);
      state = std::move(next);
    }
    const auto rollout = act(discorl::detached(state.params));
    final = objective(agent, value_fn, value, state.params, rollout, config);
    accumulate(torch::autograd::grad({final.loss + reg_loss}, meta_values, {},
                                     false, false, true),
               0);
    reg_loss = reg_loss.detach();
  } else {
    // Forward without graphs, keeping each update's inputs...
    std::vector<LearnerState> inputs;
    std::vector<Rollout> batches;
    const auto device = meta_values.front().device();
    for (int64_t k = 0; k < updates; ++k) {
      batches.push_back(next_batch(k, state.params));
      inputs.push_back(state);
      if (config.offload) {
        move(inputs.back(), torch::kCPU);
      }
      state = agent.learner_step(batches.back(), state, meta_params, false)
                  .first;
    }
    const auto rollout = act(state.params);
    LearnerState last = leaves(state);
    final = objective(agent, value_fn, value, last.params, rollout, config);
    auto adjoint = torch::autograd::grad({final.loss}, tensors_of(last), {},
                                         false, false, true);
    // ...then back through them one at a time, recomputing each: the adjoint
    // of an update's input state is its output state's, pulled back.
    for (int64_t k = updates - 1; k >= 0; --k) {
      if (config.offload) {
        move(inputs[k], device);
      }
      LearnerState input = leaves(inputs[k]);
      inputs[k] = {};
      auto [next, logs] =
          agent.learner_step(batches[k], input, meta_params, true);
      const auto reg = update_regularizer(logs, config);
      reg_loss = reg_loss + reg.detach();
      std::vector<torch::Tensor> outputs{reg}, grad_outputs{torch::ones_like(reg)};
      const auto next_tensors = tensors_of(next);
      for (size_t j = 0; j < next_tensors.size(); ++j) {
        if (adjoint[j].defined() && next_tensors[j].requires_grad()) {
          outputs.push_back(next_tensors[j]);
          grad_outputs.push_back(adjoint[j]);
        }
      }
      auto wrt = tensors_of(input);
      const size_t n_state = wrt.size();
      wrt.insert(wrt.end(), meta_values.begin(), meta_values.end());
      const auto g = torch::autograd::grad(outputs, wrt, grad_outputs, false,
                                           false, true);
      adjoint.assign(g.begin(), g.begin() + n_state);
      accumulate(g, n_state);
    }
  }
  for (size_t i = 0; i < grads.size(); ++i) {
    if (!grads[i].defined()) {
      grads[i] = torch::zeros_like(meta_values[i]);
    }
  }
  const auto loss = final.loss.detach() + reg_loss;
  Tensors logs{{"meta_loss", loss},
               {"pg_loss", final.pg_loss.detach()},
               {"reg_loss", loss - final.pg_loss.detach()},
               {"entropy", final.entropy.detach()}};
  return {loss, with_values(meta_params, grads), detached(state), final.value,
          logs};
}

namespace {

// A GPU without an index is the current one.
torch::TensorOptions with_device_index(torch::TensorOptions options) {
#ifdef DISCORL_CUDA_STREAMS
  if (options.device().is_cuda() && !options.device().has_index()) {
    return options.device(torch::Device(torch::kCUDA, c10::cuda::current_device()));
  }
#endif
  return options;
}

} // namespace

MetaTrainer::MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
                         ValueFnConfig value_config, DiscoConfig rule_config,
                         const EnvironmentFactory &make_env, Params meta_params,
                         const torch::TensorOptions &options, uint64_t seed)
    : MetaTrainer(std::move(config), std::move(agent_config),
                  std::move(value_config), std::move(rule_config),
                  TaskSuite{1,
                            [make_env](int64_t, int64_t batch_size,
                                       uint64_t env_seed, const torch::Device &) {
                              return make_env(batch_size, env_seed);
                            }},
                  std::move(meta_params), options, seed) {}

MetaTrainer::MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
                         ValueFnConfig value_config, DiscoConfig rule_config,
                         TaskSuite suite, Params meta_params,
                         const torch::TensorOptions &options, uint64_t seed)
    : config_(std::move(config)), agent_config_(std::move(agent_config)),
      value_config_(std::move(value_config)),
      rule_(std::make_shared<DiscoRule>(std::move(rule_config))),
      suite_(std::move(suite)), options_(with_device_index(options)),
      seed_(seed), rng_(seed),
      meta_params_(discorl::detached(to(meta_params, options_), true)) {
  optimizer_ = std::make_unique<torch::optim::Adam>(
      values(meta_params_), torch::optim::AdamOptions(config_.learning_rate));
  agent_adam_.learning_rate = config_.learning_rate;
  agent_adam_.max_abs_update = std::numeric_limits<double>::infinity();
  devices_.push_back(options_.device());
  if (options_.device().is_cuda()) {
    for (int64_t d = 0; static_cast<int64_t>(devices_.size()) < config_.num_devices;
         ++d) {
      if (d != options_.device().index()) {
        devices_.emplace_back(torch::kCUDA, d);
      }
    }
  }
  replicas_.resize(devices_.size());
  members_.resize(config_.num_agents);
  for (int64_t i = 0; i < config_.num_agents; ++i) {
    members_[i].device = i % devices_.size();
    agent_adam_states_.push_back(agent_adam_.init(meta_params_));
    if (config_.offload) {
      move(agent_adam_states_.back(), torch::kCPU);
    }
    start_lifetime(i);
  }
}

void MetaTrainer::start_lifetime(int64_t index) {
  Member &member = members_[index];
  const auto device = devices_[member.device];
  const c10::DeviceGuard guard(device);
  const auto options = options_.device(device);
  member.task =
      config_.num_agents >= suite_.size
          ? index % suite_.size
          : std::uniform_int_distribution<int64_t>(0, suite_.size - 1)(rng_);
  member.steps = 0;
  member.lifetime = 0;
  if (!config_.lifetimes.empty()) {
    std::vector<double> weights;
    for (const auto budget : config_.lifetimes) {
      weights.push_back(1.0 / budget);
    }
    member.lifetime = config_.lifetimes[std::discrete_distribution<size_t>(
        weights.begin(), weights.end())(rng_)];
  }
  const uint64_t seed = seed_ + 1 + lifetimes_started_++;
  if (config_.update_batch > 0) {
    const auto fresh = std::max<int64_t>(
        1, std::llround(config_.update_batch * (1 - config_.replay_fraction)));
    member.env = suite_.make(member.task, fresh, seed, device);
    member.meta_env = suite_.make(member.task, config_.meta_batch,
                                  seed + (uint64_t(1) << 32), device);
    member.meta_timestep = member.meta_env->reset();
    member.replay = std::make_unique<ReplayBuffer>(config_.replay_capacity, seed);
  } else {
    member.env = suite_.make(member.task, config_.batch_size, seed, device);
  }
  member.agent = std::make_unique<Agent>(agent_config_, rule_,
                                         member.env->observation_size(),
                                         member.env->num_actions());
  member.value_fn = std::make_unique<ValueFunction>(
      value_config_, member.env->observation_size());
  member.timestep = member.env->reset();
  member.learner = member.agent->initial_learner_state(options);
  member.value = member.value_fn->initial_state(options);
  if (config_.offload) {
    move(member.learner, torch::kCPU);
    move(member.value, torch::kCPU);
  }
}

std::vector<int64_t> MetaTrainer::tasks() const {
  std::vector<int64_t> out;
  for (const auto &member : members_) {
    out.push_back(member.task);
  }
  return out;
}

std::vector<int64_t> MetaTrainer::steps() const {
  std::vector<int64_t> out;
  for (const auto &member : members_) {
    out.push_back(member.steps);
  }
  return out;
}

std::vector<double> MetaTrainer::rewards() const {
  std::vector<double> out;
  for (const auto &member : members_) {
    out.push_back(member.reward);
  }
  return out;
}

Tensors MetaTrainer::step() {
  for (int64_t i = 0; i < static_cast<int64_t>(members_.size()); ++i) {
    if (members_[i].lifetime > 0 &&
        members_[i].steps >= members_[i].lifetime) {
      start_lifetime(i);
    }
  }
  replicate();
  return config_.update_batch > 0 ? discovery_step() : colab_step();
}

// The meta-parameters on every device, as leaves for the meta-gradients.
void MetaTrainer::replicate() {
  for (size_t d = 0; d < devices_.size(); ++d) {
    replicas_[d] = d == 0 ? meta_params_
                          : discorl::detached(
                                to(meta_params_, options_.device(devices_[d])),
                                true);
  }
}

void MetaTrainer::synchronize() const {
#ifdef DISCORL_CUDA_STREAMS
  for (const auto &device : devices_) {
    if (device.is_cuda()) {
      c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
    }
  }
#endif
}

// Disco103's meta-step: each agent updates num_inner_steps times on fresh and
// replayed trajectories and returns its meta-gradient; parallel_agents at once
// on each device.
Tensors MetaTrainer::discovery_step() {
  const int64_t n = static_cast<int64_t>(members_.size());
  std::vector<MetaGradient> results(n);
  std::vector<double> positive(n, 0.0), negative(n, 0.0);
  auto run = [&](int64_t i) {
    Member &m = members_[i];
    const auto device = devices_[m.device];
    if (config_.offload) {
      move(m.learner, device);
      move(m.value, device);
    }
    double reward = 0.0;
    const BatchFn next_batch = [&](int64_t, const Params &params) {
      const auto fresh =
          collect(*m.agent, params, *m.env, m.timestep, config_.rollout_len);
      m.steps += fresh.rewards.numel();
      reward += fresh.rewards.mean().item<double>() / config_.num_inner_steps;
      positive[i] += (fresh.rewards > 0).sum().item<double>();
      negative[i] += (fresh.rewards < 0).sum().item<double>();
      m.replay->add(config_.offload ? fresh.to(torch::kCPU) : fresh);
      const int64_t replayed = config_.update_batch - fresh.actions.size(1);
      if (replayed <= 0) {
        return fresh;
      }
      return Rollout::cat({fresh, m.replay->sample(replayed).to(device)});
    };
    const ActFn act = [&](const Params &params) {
      const auto rollout = collect(*m.agent, params, *m.meta_env,
                                   m.meta_timestep, config_.rollout_len);
      m.steps += rollout.rewards.numel();
      return rollout;
    };
    results[i] = discovery_gradient(*m.agent, *m.value_fn, replicas_[m.device],
                                    m.learner, m.value, next_batch, act,
                                    config_);
    m.learner = std::move(results[i].learner);
    m.value = std::move(results[i].value);
    m.reward = reward;
    if (config_.offload) {
      move(m.learner, torch::kCPU);
      move(m.value, torch::kCPU);
    }
  };
  run_agents(run, config_.parallel_agents);
  return apply_meta_update(
      results, std::accumulate(positive.begin(), positive.end(), 0.0),
      std::accumulate(negative.begin(), negative.end(), 0.0));
}

// Runs fn for every member: per_device worker threads on each device, each
// with its own stream and its own fixed share of the device's members, so a
// member's tensors always live on one stream. Sequential on one device with
// per_device 1, or on the CPU; true when it ran concurrently.
bool MetaTrainer::run_agents(const std::function<void(int64_t)> &fn,
                             int64_t per_device) {
  const int64_t n = static_cast<int64_t>(members_.size());
  const int64_t num_devices = static_cast<int64_t>(devices_.size());
#ifdef DISCORL_CUDA_STREAMS
  if (options_.device().is_cuda() && (per_device > 1 || num_devices > 1)) {
    // Workers' streams do not wait for the devices' current streams: finish
    // the last meta-update, the new lifetimes and the replicas first.
    synchronize();
    std::vector<std::vector<int64_t>> shares(num_devices * per_device);
    std::vector<int64_t> count(num_devices, 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t d = static_cast<int64_t>(members_[i].device);
      shares[d * per_device + count[d]++ % per_device].push_back(i);
    }
    while (streams_.size() < shares.size()) {
      const auto device = devices_[streams_.size() / per_device].index();
      streams_.push_back(c10::cuda::getStreamFromPool(false, device).unwrap());
    }
    std::vector<std::exception_ptr> errors(n);
    std::vector<std::thread> workers;
    for (size_t w = 0; w < shares.size(); ++w) {
      if (shares[w].empty()) {
        continue;
      }
      workers.emplace_back([&, w] {
        const c10::cuda::CUDAStream stream(streams_[w]);
        C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index(), true));
        c10::cuda::CUDAStreamGuard guard(stream);
        for (const int64_t i : shares[w]) {
          try {
            fn(i);
          } catch (...) {
            errors[i] = std::current_exception();
          }
        }
        stream.synchronize();
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    for (const auto &error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
    return true;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    fn(i);
  }
  return false;
}

// colabs/meta_train's meta-step: every agent acts with its current
// parameters, then learns. On the GPU the agents act concurrently, each on its
// own stream.
Tensors MetaTrainer::colab_step() {
  const int64_t n = static_cast<int64_t>(members_.size());
  std::vector<std::vector<Rollout>> train(n);
  std::vector<Rollout> valid(n);
  std::vector<double> positive(n, 0.0), negative(n, 0.0);
  auto act = [&](int64_t i) {
    Member &member = members_[i];
    member.reward = 0.0;
    for (int64_t k = 0; k < config_.num_inner_steps; ++k) {
      train[i].push_back(collect(*member.agent, member.learner.params,
                                 *member.env, member.timestep,
                                 config_.rollout_len));
      const auto &rewards = train[i].back().rewards;
      member.reward += rewards.mean().item<double>() / config_.num_inner_steps;
      positive[i] += (rewards > 0).sum().item<double>();
      negative[i] += (rewards < 0).sum().item<double>();
    }
    valid[i] = collect(*member.agent, member.learner.params, *member.env,
                       member.timestep, 2 * config_.rollout_len);
  };
  run_agents(act, n);
  std::vector<MetaGradient> results(n);
  for (int64_t i = 0; i < n; ++i) {
    Member &member = members_[i];
    const c10::DeviceGuard guard(devices_[member.device]);
    results[i] = meta_gradient(*member.agent, *member.value_fn,
                               replicas_[member.device], member.learner,
                               member.value, train[i], valid[i], config_);
    member.learner = std::move(results[i].learner);
    member.value = std::move(results[i].value);
    for (const auto &rollout : train[i]) {
      member.steps += rollout.rewards.numel();
    }
    member.steps += valid[i].rewards.numel();
  }
  return apply_meta_update(
      results, std::accumulate(positive.begin(), positive.end(), 0.0),
      std::accumulate(negative.begin(), negative.end(), 0.0));
}

Tensors MetaTrainer::apply_meta_update(std::vector<MetaGradient> &results,
                                       double positive, double negative) {
  const int64_t n = static_cast<int64_t>(results.size());
  // Each agent's meta-gradient on the main device, clipped.
  std::vector<std::vector<torch::Tensor>> agent_grads(n);
  for (int64_t i = 0; i < n; ++i) {
    agent_grads[i] = values(to(results[i].grads, options_));
    if (config_.max_grad_norm > 0) {
      torch::Tensor squared = torch::zeros({}, agent_grads[i].front().options());
      for (const auto &g : agent_grads[i]) {
        squared = squared + g.square().sum();
      }
      const auto scale =
          (config_.max_grad_norm / (squared.sqrt() + 1e-6)).clamp_max(1.0);
      for (auto &g : agent_grads[i]) {
        g = g * scale;
      }
    }
  }
  std::vector<torch::Tensor> grads = agent_grads[0];
  for (int64_t i = 1; i < n; ++i) {
    for (size_t k = 0; k < grads.size(); ++k) {
      grads[k] = grads[k] + agent_grads[i][k];
    }
  }
  Tensors logs;
  double rewards = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    for (const auto &[name, value] : results[i].logs) {
      logs[name] = logs.count(name) ? logs[name] + value : value;
    }
    rewards += members_[i].reward;
  }
  auto params = values(meta_params_);
  torch::Tensor squared_norm = torch::zeros({}, grads.front().options());
  for (size_t k = 0; k < params.size(); ++k) {
    squared_norm = squared_norm + (grads[k] / n).square().sum();
  }
  if (config_.per_agent_adam) {
    // eta += 1/n sum_i Adam_i(g_i).
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> update(params.size());
    const auto device = options_.device();
    for (int64_t i = 0; i < n; ++i) {
      if (config_.offload) {
        move(agent_adam_states_[i], device);
      }
      auto [next, state] =
          agent_adam_.step(meta_params_, with_values(meta_params_, agent_grads[i]),
                           agent_adam_states_[i]);
      agent_adam_states_[i] = std::move(state);
      if (config_.offload) {
        move(agent_adam_states_[i], torch::kCPU);
      }
      const auto next_values = values(next);
      for (size_t k = 0; k < params.size(); ++k) {
        const auto delta = next_values[k] - params[k];
        update[k] = i == 0 ? delta : update[k] + delta;
      }
    }
    for (size_t k = 0; k < params.size(); ++k) {
      params[k].add_(update[k] / n);
    }
  } else {
    for (size_t k = 0; k < params.size(); ++k) {
      params[k].mutable_grad() = grads[k] / n;
    }
    optimizer_->step();
    optimizer_->zero_grad();
  }

  for (auto &[name, value] : logs) {
    value = value / n;
  }
  logs["meta_grad_norm"] = squared_norm.sqrt();
  logs["rewards"] = torch::tensor(rewards / n);
  logs["pos_rewards"] = torch::tensor(positive / n);
  logs["neg_rewards"] = torch::tensor(negative / n);
  // Tensors from the workers' streams are freed after this: finish reading
  // them first, as those streams may reuse their memory.
  synchronize();
  return logs;
}

} // namespace discorl
