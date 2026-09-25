// Discovers an update rule, or fine-tunes one, as colabs/meta_train.ipynb.
// The rule is saved in disco_rl's format, as disco_103.npz.
//
//   discorl_meta_train out=rule.npz              # from random parameters
//   discorl_meta_train init=disco_103.npz out=rule.npz

#include <iostream>

#include "common.hpp"

using namespace discorl;

int main(int argc, char **argv) {
  apps::Flags flags(argc, argv);
  const auto device = apps::device_flag(flags);
  const auto seed = flags.integer("seed", 0);
  torch::manual_seed(seed);
  const auto options = torch::TensorOptions().device(device);

  // Defaults of colabs/meta_train.ipynb.
  MetaTrainConfig config;
  config.num_agents = flags.integer("agents", 2);
  config.rollout_len = flags.integer("rollout_len", 16);
  config.num_inner_steps = flags.integer("inner_steps", 2);
  config.batch_size = flags.integer("batch_size", 32);
  config.learning_rate = flags.real("meta_lr", 5e-4);

  AgentConfig agent;
  agent.net.dense = flags.sizes("dense", {512, 512});
  const auto model = flags.integer("model_size", 256);
  agent.net.model_head_hiddens = {model};
  agent.net.lstm_size = model;
  agent.learning_rate = flags.real("lr", 5e-4);

  ValueFnConfig value;
  value.dense = flags.sizes("value_dense", {256, 256});
  value.learning_rate = flags.real("value_lr", 1e-3);

  const auto init = flags.str("init", "");
  const auto out = flags.str("out", "rule.npz");
  const auto steps = flags.integer("steps", 800);
  const auto save_interval = flags.integer("save_interval", 100);
  const auto log_interval = flags.integer("log_interval", 10);
  const auto make_env = apps::environment_flags(flags, device, 5);
  flags.check_all_used();

  const DiscoConfig rule;
  const auto meta_params = init.empty() ? DiscoRule(rule).init_params(options)
                                        : to(load_npz(init), options);
  MetaTrainer trainer(config, agent, value, rule, make_env, meta_params,
                      options, seed);
  const auto save = [&] {
    save_npz(out, to(detached(trainer.meta_params()),
                     torch::TensorOptions(torch::kCPU)));
  };
  for (int64_t step = 1; step <= steps; ++step) {
    const auto logs = trainer.step();
    if (step % log_interval == 0) {
      apps::print_logs(step, logs);
    }
    if (step % save_interval == 0) {
      save();
    }
  }
  save();
  std::cout << "saved " << out << std::endl;
  return 0;
}
