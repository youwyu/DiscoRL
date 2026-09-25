#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "discorl/discorl.hpp"

namespace discorl::apps {

// key=value arguments; a leading "--" is optional.
class Flags {
public:
  Flags(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--", 0) == 0) {
        arg = arg.substr(2);
      }
      const auto eq = arg.find('=');
      if (eq == std::string::npos) {
        fail("expected key=value, got " + arg);
      }
      values_[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
  }

  std::string str(const std::string &key, const std::string &fallback) {
    used_.insert(key);
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }
  int64_t integer(const std::string &key, int64_t fallback) {
    return std::stoll(str(key, std::to_string(fallback)));
  }
  double real(const std::string &key, double fallback) {
    std::ostringstream s;
    s << fallback;
    return std::stod(str(key, s.str()));
  }
  // Comma-separated sizes, "512,512".
  std::vector<int64_t> sizes(const std::string &key,
                             const std::vector<int64_t> &fallback) {
    const auto it = values_.find(key);
    used_.insert(key);
    if (it == values_.end()) {
      return fallback;
    }
    std::vector<int64_t> out;
    std::istringstream s(it->second);
    for (std::string part; std::getline(s, part, ',');) {
      out.push_back(std::stoll(part));
    }
    return out;
  }
  // Rejects misspelled flags.
  void check_all_used() const {
    for (const auto &[key, value] : values_) {
      if (!used_.count(key)) {
        fail("unknown flag " + key);
      }
    }
  }

  [[noreturn]] static void fail(const std::string &message) {
    std::cerr << "error: " << message << "\n";
    std::exit(2);
  }

private:
  std::map<std::string, std::string> values_;
  std::set<std::string> used_;
};

inline torch::Device device_flag(Flags &flags) {
  return torch::Device(
      flags.str("device", torch::cuda::is_available() ? "cuda" : "cpu"));
}

// The environments by name; add new ones here.
inline EnvironmentFactory environment_flags(Flags &flags, torch::Device device,
                                            int64_t catch_size = 8) {
  const auto name = flags.str("env", "catch");
  if (name == "catch") {
    const auto rows = flags.integer("rows", catch_size);
    const auto columns = flags.integer("columns", catch_size);
    return [=](int64_t batch_size, uint64_t seed) {
      return std::make_unique<Catch>(batch_size, rows, columns, device, seed);
    };
  }
  Flags::fail("unknown env " + name);
}

inline void print_logs(int64_t step, const Tensors &logs) {
  std::cout << "step=" << step;
  for (const auto &[name, value] : logs) {
    std::cout << " " << name << "=" << value.item<double>();
  }
  std::cout << std::endl;
}

} // namespace discorl::apps
