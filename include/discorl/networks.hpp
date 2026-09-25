#pragma once

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "discorl/params.hpp"

namespace discorl {

// The agent outputs an update rule trains: heads of the state [..., size],
// and heads of the action-conditional model [..., A, size].
struct OutputSpec {
  std::vector<std::pair<std::string, int64_t>> flat;
  std::vector<std::pair<std::string, int64_t>> model;
};

// disco_rl's MLP agent (networks/nets.py) with its LSTM action model
// (networks/action_models.py).
struct NetConfig {
  std::vector<int64_t> dense{512, 512};
  double head_w_init_std = 1e-2;
  std::vector<int64_t> model_head_hiddens{128};
  int64_t lstm_size = 128;
};

class AgentNet {
public:
  AgentNet(NetConfig config, OutputSpec spec, int64_t num_actions,
           std::string name = "mlp");

  Params init(int64_t observation_size,
              const torch::TensorOptions &options) const;
  // Outputs for observations [..., D]; without `model`, only the state's.
  Tensors operator()(const Params &params, const torch::Tensor &observations,
                     bool model = true) const;

private:
  Tensors apply(const Scope &s, const torch::Tensor &observations,
                bool model) const;

  NetConfig config_;
  OutputSpec spec_; // flat heads sorted by name, as Haiku creates them
  int64_t num_actions_;
  std::string name_;
};

// The meta-network of DiscoRL's discovered rule (networks/meta_nets.py,
// LSTM with MetaLSTM). The defaults are Disco103's.
struct MetaNetConfig {
  int64_t hidden_size = 256;
  std::vector<int64_t> embedding_size{16, 1};
  int64_t prediction_size = 600;
  std::vector<int64_t> policy_channels{16, 2};
  std::vector<int64_t> policy_target_channels{16};
  double output_stddev = 0.3;
  double aux_stddev = 0.3;
  double policy_target_stddev = 0.3;
  double state_stddev = 1.0;
  // The lifetime meta-RNN.
  std::vector<int64_t> meta_policy_channels{16, 2};
  std::vector<int64_t> meta_embedding_size{16};
  std::vector<int64_t> meta_pred_embedding_size{16, 1};
  int64_t meta_hidden_size = 128;
};

// What the meta-network reads of a window (update_rules/disco.py,
// get_input_option). The logits and predictions are the agent's raw outputs.
struct MetaNetInputs {
  torch::Tensor actions;           // [T, B]
  torch::Tensor logits;            // [T+1, B, A]
  torch::Tensor behaviour_logits;  // [T+1, B, A]
  torch::Tensor target_logits;     // [T+1, B, A]
  torch::Tensor y;                 // [T+1, B, P]
  torch::Tensor target_y;          // [T+1, B, P]
  torch::Tensor z;                 // [T+1, B, A, P]
  torch::Tensor target_z;          // [T+1, B, A, P]
  torch::Tensor rewards;           // [T, B]
  torch::Tensor discounts;         // [T, B], 1 - terminal
  torch::Tensor v;                 // [T+1, B]
  torch::Tensor adv;               // [T, B]
  torch::Tensor normalized_adv;    // [T, B]
  torch::Tensor q;                 // [T+1, B, A]
  torch::Tensor qv_adv;            // [T+1, B, A]
  torch::Tensor normalized_qv_adv; // [T+1, B, A]
};

struct MetaNetOutput {
  torch::Tensor pi;             // [T, B, A] policy target logits
  torch::Tensor y;              // [T, B, P] prediction targets
  torch::Tensor z;              // [T, B, P]
  torch::Tensor meta_input_emb; // [T, B, 1]
};

struct LstmState {
  torch::Tensor h;
  torch::Tensor c;
};

class MetaNet {
public:
  explicit MetaNet(MetaNetConfig config = {});

  Params init(const torch::TensorOptions &options) const;
  LstmState initial_state(const torch::TensorOptions &options) const;
  // The targets for a window, and the lifetime meta-RNN's next state.
  std::pair<MetaNetOutput, LstmState> operator()(const Params &params,
                                                 const MetaNetInputs &in,
                                                 const LstmState &state) const;

private:
  std::pair<MetaNetOutput, LstmState>
  apply(const Scope &s, const MetaNetInputs &in, const LstmState &state) const;

  MetaNetConfig config_;
};

} // namespace discorl
