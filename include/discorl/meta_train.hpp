#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "discorl/agent.hpp"
#include "discorl/env.hpp"
#include "discorl/replay.hpp"
#include "discorl/update_rules.hpp"

namespace discorl {

// Meta-training of the discovered rule: agents update with the rule, and the
// rule follows the meta-gradient of the updated agents' policy-gradient loss.
// colabs/meta_train.ipynb's loop by default, Disco103's with update_batch.
struct MetaTrainConfig {
  int64_t num_agents = 2;
  int64_t rollout_len = 16;
  int64_t num_inner_steps = 2;
  int64_t batch_size = 32;
  // Disco103's discovery (Oh et al. 2025, Methods): each agent update takes
  // `update_batch` trajectories, `replay_fraction` of them from a replay
  // buffer; each meta-step backpropagates through num_inner_steps updates and
  // generates `meta_batch` trajectories with the updated agent for the
  // meta-objective and the meta-value function. 0 keeps colabs/meta_train's
  // loop: batch_size fresh trajectories per update, a validation rollout
  // acted before the updates.
  int64_t update_batch = 0;
  double replay_fraction = 0.9;
  int64_t replay_capacity = 4096; // trajectories
  int64_t meta_batch = 48;
  // Backpropagate through one agent update at a time, recomputing it from
  // stored inputs: memory for one update instead of all of them.
  bool recompute = true;
  // Agents computing their meta-gradients at once on the GPU, each on its own
  // stream and with memory for one update's graph.
  int64_t parallel_agents = 1;
  // The population split over `processes` processes, this one `rank`: each
  // runs the agents rank, rank + processes, ..., and after every meta-step
  // they exchange their meta-updates through files in `exchange_dir`, where
  // rank 0 applies the average.
  int64_t processes = 1, rank = 0;
  std::string exchange_dir;
  double exchange_timeout = 3600; // s to wait for the other processes
  // Keep each agent's state and replay in host memory between its turns, and
  // the inputs of its updates until the backward pass reaches them, so the
  // GPU holds only the agents in flight.
  bool offload = false;
  // Environment steps an agent lives before it restarts from new parameters:
  // drawn from these budgets with probability inversely proportional to the
  // budget, as Disco103's 200M, 100M, 50M and 20M. Empty keeps agents forever.
  std::vector<int64_t> lifetimes;
  double learning_rate = 5e-4; // Adam on the meta-parameters
  // Disco103's meta-optimizer: a separate Adam per agent on its meta-gradient,
  // then the updates averaged. Otherwise one Adam on the averaged gradient.
  bool per_agent_adam = false;
  double max_grad_norm = 0.0; // clips each meta-gradient; 0 does not
  // Meta-loss regularizers.
  double entropy_cost = 1e-2;            // the validation policy's entropy
  double prediction_entropy_cost = 1e-3; // y and z entropies
  double target_mean_cost = 1e-3;        // squared means of the targets
  double target_kl_cost = 1e-2;          // KL(target policy || pi target)
};

struct MetaGradient {
  torch::Tensor loss;
  Params grads;
  LearnerState learner; // after the inner updates, without history
  ValueState value;
  Tensors logs;
};

// The meta-gradient of one agent: inner updates on `train`, each rollout
// [L, B], then the meta-loss on `valid`, [2L, B], acted before the updates.
MetaGradient meta_gradient(const Agent &agent, const ValueFunction &value_fn,
                           const Params &meta_params,
                           const LearnerState &learner, const ValueState &value,
                           const std::vector<Rollout> &train,
                           const Rollout &valid, const MetaTrainConfig &config);

// The meta-gradient of Disco103's discovery for one agent: it updates
// num_inner_steps times on batches from `next_batch` (given the update's index
// and current parameters), then `act` generates a rollout with the updated
// parameters for the advantage actor-critic meta-objective.
using BatchFn = std::function<Rollout(int64_t update, const Params &params)>;
using ActFn = std::function<Rollout(const Params &params)>;
MetaGradient discovery_gradient(const Agent &agent,
                                const ValueFunction &value_fn,
                                const Params &meta_params,
                                const LearnerState &learner,
                                const ValueState &value,
                                const BatchFn &next_batch, const ActFn &act,
                                const MetaTrainConfig &config);

using EnvironmentFactory = std::function<std::unique_ptr<Environment>(
    int64_t batch_size, uint64_t seed)>;

// Environments to discover a rule on, by task index; tasks may differ in
// observations and actions.
struct TaskSuite {
  int64_t size = 1;
  std::function<std::unique_ptr<Environment>(
      int64_t task, int64_t batch_size, uint64_t seed,
      const torch::Device &device)>
      make;
};

class MetaTrainer {
public:
  MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
              ValueFnConfig value_config, DiscoConfig rule_config,
              const EnvironmentFactory &make_env, Params meta_params,
              const torch::TensorOptions &options, uint64_t seed = 0);
  // Members take tasks in order, keeping them (Disco57 cycles through Atari);
  // with fewer members than tasks, each draws one per lifetime.
  MetaTrainer(MetaTrainConfig config, AgentConfig agent_config,
              ValueFnConfig value_config, DiscoConfig rule_config,
              TaskSuite suite, Params meta_params,
              const torch::TensorOptions &options, uint64_t seed = 0);

  // One meta-update across the population.
  Tensors step();
  const Params &meta_params() const { return meta_params_; }
  // Each member's task, the experience its agent has used, and its mean
  // reward over the last meta-update.
  std::vector<int64_t> tasks() const;
  std::vector<int64_t> steps() const;
  std::vector<double> rewards() const;

private:
  struct Member {
    int64_t task = 0, steps = 0, lifetime = 0; // steps: experience used
    double reward = 0.0;
    int64_t index = 0;     // in the whole population
    int64_t lifetimes = 0; // started
    std::unique_ptr<Environment> env;
    std::unique_ptr<Environment> meta_env; // Disco103: meta_batch envs
    TimeStep meta_timestep;
    std::unique_ptr<ReplayBuffer> replay;
    std::unique_ptr<Agent> agent;
    std::unique_ptr<ValueFunction> value_fn;
    TimeStep timestep;
    LearnerState learner;
    ValueState value;
  };

  void start_lifetime(int64_t index);
  Tensors discovery_step();
  Tensors colab_step();
  bool run_agents(const std::function<void(int64_t)> &fn, int64_t threads);
  void synchronize() const;
  void exchange(std::vector<torch::Tensor> &grads,
                std::vector<torch::Tensor> &updates, Tensors &logs,
                std::vector<double> &totals);
  Tensors apply_meta_update(std::vector<MetaGradient> &results, double positive,
                            double negative);

  MetaTrainConfig config_;
  AgentConfig agent_config_;
  ValueFnConfig value_config_;
  std::shared_ptr<const DiscoRule> rule_;
  TaskSuite suite_;
  torch::TensorOptions options_;
  uint64_t seed_;
  std::mt19937_64 rng_;
  Params meta_params_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
  Optimizer agent_adam_;              // per-agent meta-optimizer
  std::vector<AdamState> agent_adam_states_;
  std::vector<c10::Stream> streams_;  // one per worker, kept across steps
  int64_t exchanges_ = 0;
  std::vector<Member> members_;
};

} // namespace discorl
