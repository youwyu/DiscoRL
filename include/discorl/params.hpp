#pragma once

#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace discorl {

// Named tensors. As parameters, names follow Haiku ("<module>/w"), which makes
// them interchangeable with google-deepmind/disco_rl: its disco_103.npz loads
// here, and rules discovered here load there.
using Tensors = std::map<std::string, torch::Tensor>;
using Params = Tensors;

std::vector<torch::Tensor> values(const Params &params);
// `like`'s names with new values, in the order of values().
Params with_values(const Params &like,
                   const std::vector<torch::Tensor> &values);
// Leaves without history, optionally requiring gradients.
Params detached(const Params &params, bool requires_grad = false);
Params to(const Params &params, const torch::TensorOptions &options);
// Entries under "<prefix>/", with the prefix removed.
Params subtree(const Tensors &tensors, const std::string &prefix);

// Haiku's TruncatedNormal: N(0, stddev^2) truncated to two deviations.
torch::Tensor truncated_normal(at::IntArrayRef shape, double stddev,
                               const torch::TensorOptions &options);

// Parameter access for networks written once for creation and use, as in
// Haiku's transform: a creating scope initializes a name on its first get().
class Scope {
public:
  explicit Scope(const Params &params);
  Scope(Params &params, torch::TensorOptions options);

  Scope operator/(const std::string &name) const;
  // Created from a truncated normal with `stddev`; 0 gives zeros.
  torch::Tensor get(const std::string &name, at::IntArrayRef shape,
                    double stddev) const;

private:
  const Params *params_;
  Params *created_ = nullptr;
  torch::TensorOptions options_;
  std::string path_;
};

// numpy .npz archives of float32/64 or int32/64 arrays.
Tensors load_npz(const std::string &path);
void save_npz(const std::string &path, const Tensors &arrays);

} // namespace discorl
