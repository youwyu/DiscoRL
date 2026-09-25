// Trains an agent by an update rule, as colabs/eval.ipynb: acting envs feed a
// replay buffer of trajectories, and each step updates on a sampled batch.
//
//   discorl_train rule=disco rule_params=disco_103.npz
//   discorl_train rule=actor_critic

#include <deque>
#include <iostream>
#include <numeric>

#include "common.hpp"
#include "discorl/replay.hpp"

using namespace discorl;

int main(int argc, char **argv) {
  apps::Flags flags(argc, argv);
  const auto device = apps::device_flag(flags);
  const auto seed = flags.integer("seed", 0);
  torch::manual_seed(seed);
  const auto options = torch::TensorOptions().device(device);

  const auto rule_name = flags.str("rule", "disco");
  const bool disco = rule_name == "disco";
  std::shared_ptr<const UpdateRule> rule;
  Params meta_params;
  if (disco) {
    rule = std::make_shared<DiscoRule>();
    const auto path = flags.str("rule_params", "");
    if (path.empty()) {
      apps::Flags::fail("rule=disco needs rule_params=<rule.npz>, e.g. "
                        "disco_103.npz or one from discorl_meta_train");
    }
    meta_params = to(load_npz(path), options);
  } else if (rule_name == "actor_critic") {
    rule = std::make_shared<ActorCritic>();
  } else {
    apps::Flags::fail("unknown rule " + rule_name);
  }

  // Defaults of colabs/eval.ipynb and agent.get_settings_actor_critic.
  AgentConfig config;
  config.net.dense =
      flags.sizes("dense", disco ? std::vector<int64_t>{512, 512}
                                 : std::vector<int64_t>{64, 32, 32});
  const auto model = flags.integer("model_size", disco ? 128 : 64);
  config.net.model_head_hiddens = {model};
  config.net.lstm_size = model;
  config.learning_rate = flags.real("lr", disco ? 1e-2 : 5e-4);

  const auto steps = flags.integer("steps", 1000);
  const auto batch_size = flags.integer("batch_size", 64);
  const auto rollout_len = flags.integer("rollout_len", 29);
  const auto replay_ratio = flags.integer("replay_ratio", 32);
  const auto capacity = flags.integer("buffer_size", 1024);
  const auto log_interval = flags.integer("log_interval", 50);
  const auto make_env = apps::environment_flags(flags, device);
  flags.check_all_used();

  const auto num_envs = std::max<int64_t>(1, batch_size / replay_ratio);
  auto env = make_env(num_envs, seed);
  const Agent agent(config, rule, env->observation_size(), env->num_actions());
  auto state = agent.initial_learner_state(options);
  ReplayBuffer buffer(capacity, seed);
  auto timestep = env->reset();

  std::vector<double> running(num_envs, 0.0);
  std::deque<double> returns; // the latest episodes
  int64_t env_steps = 0;
  Tensors logs;
  for (int64_t step = 1; step <= steps; ++step) {
    const auto rollout =
        collect(agent, state.params, *env, timestep, rollout_len);
    buffer.add(rollout);
    env_steps += rollout.rewards.numel();
    const auto rewards = rollout.rewards.cpu();
    const auto discounts = rollout.discounts.cpu();
    for (int64_t t = 0; t < rewards.size(0); ++t) {
      for (int64_t b = 0; b < num_envs; ++b) {
        running[b] += rewards[t][b].item<double>();
        if (discounts[t][b].item<double>() == 0.0) {
          returns.push_back(running[b]);
          running[b] = 0.0;
          if (returns.size() > 100) {
            returns.pop_front();
          }
        }
      }
    }
    if (static_cast<int64_t>(buffer.size()) >= batch_size) {
      auto [next, step_logs] = agent.learner_step(buffer.sample(batch_size),
                                                  state, meta_params, false);
      state = std::move(next);
      logs = std::move(step_logs);
    }
    if (step % log_interval == 0) {
      const double mean_return =
          returns.empty()
              ? 0.0
              : std::accumulate(returns.begin(), returns.end(), 0.0) /
                    returns.size();
      std::cout << "step=" << step << " env_steps=" << env_steps
                << " avg_return=" << mean_return;
      if (logs.count("total_loss")) {
        std::cout << " total_loss=" << logs.at("total_loss").item<double>();
      }
      std::cout << std::endl;
    }
  }
  return 0;
}
