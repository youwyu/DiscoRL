#include "discorl/params.hpp"

#include <zlib.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace discorl {
namespace {

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error("discorl: " + message);
  }
}

uint32_t read_le(const std::string &bytes, size_t at, int width) {
  check(at + width <= bytes.size(), "truncated archive");
  uint32_t value = 0;
  for (int i = width - 1; i >= 0; --i) {
    value = value << 8 | static_cast<uint8_t>(bytes[at + i]);
  }
  return value;
}

void write_le(std::string &bytes, uint32_t value, int width) {
  for (int i = 0; i < width; ++i) {
    bytes.push_back(static_cast<char>(value >> (8 * i) & 0xff));
  }
}

std::string inflate_raw(const std::string &zip, size_t at, size_t packed,
                        size_t size) {
  check(at + packed <= zip.size(), "truncated archive");
  std::string out(size, '\0');
  z_stream stream{};
  stream.next_in =
      reinterpret_cast<Bytef *>(const_cast<char *>(zip.data() + at));
  stream.avail_in = static_cast<uInt>(packed);
  stream.next_out = reinterpret_cast<Bytef *>(out.data());
  stream.avail_out = static_cast<uInt>(size);
  const bool ok = inflateInit2(&stream, -MAX_WBITS) == Z_OK &&
                  inflate(&stream, Z_FINISH) == Z_STREAM_END &&
                  stream.total_out == size;
  inflateEnd(&stream);
  check(ok, "corrupt archive");
  return out;
}

struct NpyType {
  const char *descr;
  torch::ScalarType type;
};
constexpr NpyType kNpyTypes[] = {{"<f4", torch::kFloat32},
                                 {"<f8", torch::kFloat64},
                                 {"<i4", torch::kInt32},
                                 {"<i8", torch::kInt64}};

// A little-endian array, in C or Fortran order.
torch::Tensor parse_npy(const std::string &npy, const std::string &name) {
  check(npy.compare(0, 6, "\x93NUMPY") == 0, name + " is not a .npy array");
  const bool v1 = npy[6] == 1;
  const size_t header_at = v1 ? 10 : 12;
  const size_t header_len = read_le(npy, 8, v1 ? 2 : 4);
  const auto header = npy.substr(header_at, header_len);
  const NpyType *type = nullptr;
  for (const auto &candidate : kNpyTypes) {
    if (header.find(std::string("'") + candidate.descr + "'") !=
        std::string::npos) {
      type = &candidate;
    }
  }
  check(type != nullptr, name + " has an unsupported dtype");
  const bool fortran =
      header.find("'fortran_order': True") != std::string::npos;
  const auto open = header.find('(', header.find("'shape'"));
  const auto close = header.find(')', open);
  std::vector<int64_t> shape;
  std::istringstream dims(header.substr(open + 1, close - open - 1));
  for (std::string dim; std::getline(dims, dim, ',');) {
    if (dim.find_first_not_of(' ') != std::string::npos) {
      shape.push_back(std::stoll(dim));
    }
  }
  // Fortran order is C order of the reversed shape.
  const auto stored =
      fortran ? std::vector<int64_t>(shape.rbegin(), shape.rend()) : shape;
  auto tensor = torch::empty(stored, type->type);
  const size_t data = header_at + header_len;
  const size_t bytes = tensor.numel() * tensor.element_size();
  check(npy.size() == data + bytes, name + " is truncated");
  std::memcpy(tensor.data_ptr(), npy.data() + data, bytes);
  if (fortran) {
    std::vector<int64_t> reverse(shape.size());
    for (size_t i = 0; i < reverse.size(); ++i) {
      reverse[i] = static_cast<int64_t>(reverse.size() - 1 - i);
    }
    tensor = tensor.permute(reverse).contiguous();
  }
  return tensor;
}

std::string format_npy(const torch::Tensor &array) {
  const auto t = array.detach().cpu().contiguous();
  const NpyType *type = nullptr;
  for (const auto &candidate : kNpyTypes) {
    if (candidate.type == t.scalar_type()) {
      type = &candidate;
    }
  }
  check(type != nullptr,
        "cannot save dtype " + std::string(c10::toString(t.scalar_type())));
  std::string shape; // "()", "(3,)", "(2, 3)"
  for (int64_t i = 0; i < t.dim(); ++i) {
    shape += std::to_string(t.size(i));
    shape += i + 1 < t.dim() ? ", " : (t.dim() == 1 ? "," : "");
  }
  std::string header = std::string("{'descr': '") + type->descr +
                       "', 'fortran_order': False, 'shape': (" + shape + "), }";
  // The data starts 64-byte aligned, after a newline.
  header.append(63 - (10 + header.size()) % 64, ' ');
  header.push_back('\n');
  std::string npy("\x93NUMPY\x01\x00", 8);
  write_le(npy, static_cast<uint32_t>(header.size()), 2);
  npy += header;
  npy.append(static_cast<const char *>(t.data_ptr()),
             t.numel() * t.element_size());
  return npy;
}

} // namespace

std::vector<torch::Tensor> values(const Params &params) {
  std::vector<torch::Tensor> out;
  out.reserve(params.size());
  for (const auto &[name, value] : params) {
    out.push_back(value);
  }
  return out;
}

Params with_values(const Params &like,
                   const std::vector<torch::Tensor> &values) {
  check(like.size() == values.size(), "parameter count mismatch");
  Params out;
  size_t i = 0;
  for (const auto &[name, value] : like) {
    out[name] = values[i++];
  }
  return out;
}

Params detached(const Params &params, bool requires_grad) {
  Params out;
  for (const auto &[name, value] : params) {
    out[name] = value.detach().requires_grad_(requires_grad);
  }
  return out;
}

Params to(const Params &params, const torch::TensorOptions &options) {
  Params out;
  for (const auto &[name, value] : params) {
    out[name] = value.to(options);
  }
  return out;
}

Params subtree(const Tensors &tensors, const std::string &prefix) {
  Params out;
  const auto start = prefix + "/";
  for (const auto &[name, value] : tensors) {
    if (name.compare(0, start.size(), start) == 0) {
      out[name.substr(start.size())] = value;
    }
  }
  return out;
}

torch::Tensor truncated_normal(at::IntArrayRef shape, double stddev,
                               const torch::TensorOptions &options) {
  torch::NoGradGuard no_grad;
  if (stddev == 0.0) {
    return torch::zeros(shape, options);
  }
  // Inverse CDF of the standard normal on [Phi(-2), Phi(2)].
  const double bound = std::erf(2.0 / std::sqrt(2.0));
  auto x = torch::empty(shape, options.dtype(torch::kFloat64));
  x.uniform_(-bound, bound).erfinv_().mul_(std::sqrt(2.0) * stddev);
  return torch::empty(shape, options).copy_(x);
}

Scope::Scope(const Params &params) : params_(&params) {}

Scope::Scope(Params &params, torch::TensorOptions options)
    : params_(&params), created_(&params), options_(std::move(options)) {}

Scope Scope::operator/(const std::string &name) const {
  Scope child = *this;
  child.path_ = path_.empty() ? name : path_ + "/" + name;
  return child;
}

torch::Tensor Scope::get(const std::string &name, at::IntArrayRef shape,
                         double stddev) const {
  const auto key = path_.empty() ? name : path_ + "/" + name;
  auto it = params_->find(key);
  if (it == params_->end()) {
    check(created_ != nullptr, "missing parameter " + key);
    it =
        created_->emplace(key, truncated_normal(shape, stddev, options_)).first;
  }
  check(it->second.sizes() == shape, key + " has shape " +
                                         c10::str(it->second.sizes()) +
                                         ", expected " + c10::str(shape));
  return it->second;
}

Tensors load_npz(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  check(file.good(), "cannot read " + path);
  const std::string zip((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
  const size_t end = zip.rfind(std::string("PK\x05\x06", 4));
  check(end != std::string::npos, path + " is not a zip archive");
  Tensors arrays;
  size_t entry = read_le(zip, end + 16, 4);
  for (uint32_t i = 0, n = read_le(zip, end + 10, 2); i < n; ++i) {
    check(read_le(zip, entry, 4) == 0x02014b50, "corrupt archive " + path);
    const uint32_t method = read_le(zip, entry + 10, 2);
    const size_t packed = read_le(zip, entry + 20, 4);
    const size_t size = read_le(zip, entry + 24, 4);
    const size_t name_len = read_le(zip, entry + 28, 2);
    const size_t local = read_le(zip, entry + 42, 4);
    auto name = zip.substr(entry + 46, name_len);
    entry += 46 + name_len + read_le(zip, entry + 30, 2) +
             read_le(zip, entry + 32, 2);
    // Sizes come from here: numpy writes zip64 local headers.
    check((method == 0 || method == 8) && packed != 0xffffffff,
          "unsupported zip entry " + name);
    const size_t data =
        local + 30 + read_le(zip, local + 26, 2) + read_le(zip, local + 28, 2);
    const auto npy = method == 0 ? zip.substr(data, size)
                                 : inflate_raw(zip, data, packed, size);
    name.resize(name.size() - 4); // ".npy"
    arrays[name] = parse_npy(npy, name);
  }
  return arrays;
}

void save_npz(const std::string &path, const Tensors &arrays) {
  std::string zip;
  std::string directory;
  for (const auto &[name, array] : arrays) {
    const auto npy = format_npy(array);
    const auto file = name + ".npy";
    const auto crc = static_cast<uint32_t>(
        crc32(0L, reinterpret_cast<const Bytef *>(npy.data()),
              static_cast<uInt>(npy.size())));
    const auto offset = static_cast<uint32_t>(zip.size());
    const auto size = static_cast<uint32_t>(npy.size());
    auto header = [&](std::string &out, bool central) {
      write_le(out, central ? 0x02014b50 : 0x04034b50, 4);
      if (central) {
        write_le(out, 20, 2); // made by
      }
      write_le(out, 20, 2); // version needed
      write_le(out, 0, 2);  // flags
      write_le(out, 0, 2);  // stored
      write_le(out, 0, 4);  // time, date
      write_le(out, crc, 4);
      write_le(out, size, 4);
      write_le(out, size, 4);
      write_le(out, static_cast<uint32_t>(file.size()), 2);
      write_le(out, 0, 2); // extra
      if (central) {
        write_le(out, 0, 2); // comment
        write_le(out, 0, 2); // disk
        write_le(out, 0, 2); // internal attributes
        write_le(out, 0, 4); // external attributes
        write_le(out, offset, 4);
      }
      out += file;
    };
    header(zip, false);
    zip += npy;
    header(directory, true);
  }
  const auto directory_at = static_cast<uint32_t>(zip.size());
  zip += directory;
  write_le(zip, 0x06054b50, 4);
  write_le(zip, 0, 4); // disks
  write_le(zip, static_cast<uint32_t>(arrays.size()), 2);
  write_le(zip, static_cast<uint32_t>(arrays.size()), 2);
  write_le(zip, static_cast<uint32_t>(directory.size()), 4);
  write_le(zip, directory_at, 4);
  write_le(zip, 0, 2); // comment
  std::ofstream out(path, std::ios::binary);
  check(out.good(), "cannot write " + path);
  out.write(zip.data(), static_cast<std::streamsize>(zip.size()));
}

} // namespace discorl
