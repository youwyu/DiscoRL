#include "discorl/nn.hpp"

#include <cmath>
#include <string>

namespace discorl::nn {

torch::Tensor linear(const Scope &s, const torch::Tensor &x, int64_t size,
                     double stddev) {
  const int64_t fan_in = x.size(-1);
  const auto w =
      s.get("w", {fan_in, size}, stddev < 0 ? 1.0 / std::sqrt(fan_in) : stddev);
  const auto b = s.get("b", {size}, 0.0);
  return torch::matmul(x, w) + b;
}

torch::Tensor mlp(const Scope &s, torch::Tensor x,
                  const std::vector<int64_t> &sizes, double stddev) {
  for (size_t i = 0; i < sizes.size(); ++i) {
    x = linear(s / ("~/linear_" + std::to_string(i)), x, sizes[i], stddev);
    if (i + 1 < sizes.size()) {
      x = torch::relu(x);
    }
  }
  return x;
}

std::pair<torch::Tensor, torch::Tensor> lstm(const Scope &s,
                                             const torch::Tensor &x,
                                             const torch::Tensor &h,
                                             const torch::Tensor &c) {
  const auto g =
      linear(s / "linear", torch::cat({x, h}, -1), 4 * h.size(-1)).chunk(4, -1);
  const auto cell =
      torch::sigmoid(g[2] + 1.0) * c + torch::sigmoid(g[0]) * torch::tanh(g[1]);
  return {torch::sigmoid(g[3]) * torch::tanh(cell), cell};
}

torch::Tensor action_conv(const Scope &s, torch::Tensor x,
                          const std::vector<int64_t> &channels) {
  for (size_t i = 0; i < channels.size(); ++i) {
    const auto name = i == 0 ? "conv1_d" : "conv1_d_" + std::to_string(i);
    x = torch::cat({x, x.mean(-2, true).expand_as(x)}, -1);
    // Conv1D weights are [kernel, in, out]; the default fan_in is kernel * in.
    const int64_t fan_in = x.size(-1);
    const auto w =
        (s / name).get("w", {1, fan_in, channels[i]}, 1.0 / std::sqrt(fan_in));
    const auto b = (s / name).get("b", {channels[i]}, 0.0);
    x = torch::relu(torch::matmul(x, w[0]) + b);
  }
  return x;
}

torch::Tensor take(const torch::Tensor &table, const torch::Tensor &index) {
  const int64_t dim = index.dim();
  auto idx = index.to(torch::kLong);
  for (int64_t i = dim; i < table.dim(); ++i) {
    idx = idx.unsqueeze(-1);
  }
  auto shape = table.sizes().vec();
  shape[dim] = 1;
  return table.gather(dim, idx.expand(shape)).squeeze(dim);
}

torch::Tensor log_prob(const torch::Tensor &logits,
                       const torch::Tensor &index) {
  return nn::take(logits.log_softmax(-1), index);
}

torch::Tensor entropy(const torch::Tensor &logits) {
  const auto log_p = logits.log_softmax(-1);
  return -(log_p.exp() * log_p).sum(-1);
}

torch::Tensor kl(const torch::Tensor &p, const torch::Tensor &q) {
  const auto log_p = p.log_softmax(-1);
  return (log_p.exp() * (log_p - q.log_softmax(-1))).sum(-1);
}

} // namespace discorl::nn
