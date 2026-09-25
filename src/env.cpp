#include "discorl/env.hpp"

#include <ATen/CPUGeneratorImpl.h>

#include <vector>

namespace discorl {

Catch::Catch(int64_t batch_size, int64_t rows, int64_t columns,
             torch::Device device, uint64_t seed)
    : batch_size_(batch_size), rows_(rows), columns_(columns), device_(device),
      generator_(at::detail::createCPUGenerator(seed)) {}

torch::Tensor Catch::new_balls(int64_t n) {
  return torch::randint(0, columns_, {n}, generator_, torch::kLong).to(device_);
}

TimeStep Catch::reset() {
  const auto longs = torch::TensorOptions().dtype(torch::kLong).device(device_);
  ball_y_ = torch::zeros({batch_size_}, longs);
  ball_x_ = new_balls(batch_size_);
  paddle_x_ = torch::full({batch_size_}, columns_ / 2, longs);
  const auto zeros =
      torch::zeros({batch_size_}, torch::TensorOptions().device(device_));
  return {render(), zeros, zeros};
}

TimeStep Catch::step(const torch::Tensor &actions) {
  paddle_x_ = (paddle_x_ + actions.to(torch::kLong) - 1).clamp(0, columns_ - 1);
  ball_y_ = ball_y_ + 1;
  const auto last = ball_y_ == rows_ - 1;
  const auto caught = ball_x_ == paddle_x_;
  const auto reward =
      last.to(torch::kFloat32) * (2.0 * caught.to(torch::kFloat32) - 1.0);
  // Finished episodes restart right away.
  ball_y_ = torch::where(last, torch::zeros_like(ball_y_), ball_y_);
  ball_x_ = torch::where(last, new_balls(batch_size_), ball_x_);
  paddle_x_ =
      torch::where(last, torch::full_like(paddle_x_, columns_ / 2), paddle_x_);
  return {render(), reward, last.to(torch::kFloat32)};
}

torch::Tensor Catch::render() const {
  auto board = torch::zeros({batch_size_, rows_ * columns_},
                            torch::TensorOptions().device(device_));
  const auto rows = torch::arange(batch_size_, ball_y_.options());
  board.index_put_({rows, ball_y_ * columns_ + ball_x_}, 1.0);
  board.index_put_(
      {rows, torch::full_like(paddle_x_, rows_ - 1) * columns_ + paddle_x_},
      1.0);
  return board;
}

Rollout collect(const Agent &agent, const Params &params, Environment &env,
                TimeStep &timestep, int64_t length) {
  std::vector<torch::Tensor> observations, actions, rewards, discounts, logits;
  const auto options = params.begin()->second.options().requires_grad(false);
  for (int64_t t = 0; t < length; ++t) {
    const auto observation = timestep.observation.to(options);
    auto [action, policy] = agent.act(params, observation);
    observations.push_back(observation);
    actions.push_back(action);
    rewards.push_back(timestep.reward.to(options));
    discounts.push_back(1.0 - timestep.last.to(options));
    logits.push_back(policy);
    timestep = env.step(action);
  }
  Rollout rollout{torch::stack(observations),
                  torch::stack(actions),
                  torch::stack(rewards),
                  torch::stack(discounts),
                  {{"logits", torch::stack(logits)}}};
  return rollout;
}

} // namespace discorl
