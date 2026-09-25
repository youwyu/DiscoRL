#pragma once

#include <utility>
#include <vector>

#include <torch/torch.h>

#include "discorl/params.hpp"

// Haiku's layers as functions of a Scope, and categorical distributions.
namespace discorl::nn {

// hk.Linear: x w + b, w ~ TruncatedNormal(stddev), 1/sqrt(fan_in) if < 0.
torch::Tensor linear(const Scope &s, const torch::Tensor &x, int64_t size,
                     double stddev = -1.0);
// hk.nets.MLP: layers "~/linear_i", ReLU between them.
torch::Tensor mlp(const Scope &s, torch::Tensor x,
                  const std::vector<int64_t> &sizes, double stddev = -1.0);
// hk.LSTM step: gates [i, g, f, o] from [x, h], forget bias 1.
std::pair<torch::Tensor, torch::Tensor> lstm(const Scope &s,
                                             const torch::Tensor &x,
                                             const torch::Tensor &h,
                                             const torch::Tensor &c);
// disco_rl's kernel-1 Conv1D blocks over the action axis of [..., A, C];
// each block also sees the mean over actions.
torch::Tensor action_conv(const Scope &s, torch::Tensor x,
                          const std::vector<int64_t> &channels);

// table[i, j, index[i, j], ...] over index's dims (utils.batch_lookup).
torch::Tensor take(const torch::Tensor &table, const torch::Tensor &index);
torch::Tensor log_prob(const torch::Tensor &logits, const torch::Tensor &index);
torch::Tensor entropy(const torch::Tensor &logits);
// KL(softmax(p) || softmax(q)).
torch::Tensor kl(const torch::Tensor &p, const torch::Tensor &q);

} // namespace discorl::nn
