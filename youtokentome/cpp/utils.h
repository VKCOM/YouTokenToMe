#pragma once

#include <iostream>
#include <istream>
#include <string>
#include <vector>

namespace vkcom {

enum OutputType { ID, SUBWORD };

struct DecodeResult {
  std::vector<int> ids;
  std::vector<std::string> pieces;
};

struct Status {
  int code{0};
  std::string message;
  Status() = default;
  Status(int code, std::string message);

  const std::string &error_message() const;
  bool ok() const;
};

struct SpecialTokens {
  int pad_id = -1;
  int unk_id = -1;
  int bos_id = -1;
  int eos_id = -1;

  SpecialTokens() = default;

  SpecialTokens(int pad_id, int unk_id, int bos_id, int eos_id);

  void dump(std::ofstream &fout);

  void load(std::ifstream &fin);

  uint32_t max_id() const;

  bool taken_id(int id) const;

  uint64_t n_special_tokens() const;
};

std::vector<std::string> read_all_lines(std::istream& stream);

std::vector<std::string> read_lines(std::istream& stream, uint64_t batch_limit, uint64_t *processed);

Status fast_read_file_utf8(const std::string &file_name, std::string *file_content);

template <typename T>
void write_to_stdout(const std::vector<T> &items, bool flush) {
  for (const auto &item : items) {
    std::cout << item << " ";
  }
  std::cout << "\n";
  if (flush) {
    std::cout << std::flush;
  }
}

template <typename T>
void write_to_stdout(const std::vector<std::vector<T>> &sentences, bool flush) {
  for (const auto &sentence : sentences) {
    write_to_stdout(sentence, false);
  }
  if (flush) {
    std::cout << std::flush;
  }
}

template <typename T>
struct VectorSegment {
 private:
  const T *begin_;
  const T *end_;
  uint64_t hash_;

 public:
  VectorSegment(const T *begin, const T *end, uint64_t hash)
   : begin_(begin), end_(end), hash_(hash) {}

  bool operator==(const VectorSegment &other) const {
    if (other.hash() != hash() || end_ - begin_ != other.end_ - other.begin_) {
      return false;
    }
    for (auto it = begin_, other_it = other.begin_; it != end_; it++, other_it++) {
      if (*it != *other_it) {
        return false;
      }
    }
    return true;
  }

  uint64_t hash() const { return hash_; }
};

template <typename T>
class VectorSegmentBuilder {
 private:
  constexpr static uint64_t MOD = 2032191299;
  constexpr static uint64_t P = 726328703;

  const T *begin_;
  const T *end_;
  std::vector<uint64_t> prefix_hash_;

 public:
  explicit VectorSegmentBuilder(const std::vector<T> &segment)
   : VectorSegmentBuilder(segment.data(), segment.data() + segment.size()) {}

  VectorSegmentBuilder(const T *begin, const T *end) : begin_(begin), end_(end) {
    using HashT = typename std::make_unsigned<T>::type;
    uint64_t hash = 0;
    prefix_hash_.reserve(static_cast<size_t>(end - begin));
    for (const T *it = begin_; it != end_; it++) {
      hash = (hash * P + static_cast<HashT>(*it)) % MOD;
      prefix_hash_.push_back(hash);
    }
  }

  VectorSegment<T> finish() const { return VectorSegment<T>(begin_, end_, hash()); }

  size_t size() const { return prefix_hash_.size(); }

  bool empty() const { return prefix_hash_.empty(); }

  uint64_t hash() const { return prefix_hash_.empty() ? 0 : prefix_hash_.back(); }

  void pop_back() noexcept {
    if (!prefix_hash_.empty()) {
      prefix_hash_.pop_back();
      --end_;
    }
  }
};

using BpeVectorSegment = VectorSegment<char>;
using WordPieceVectorSegment = VectorSegment<uint32_t>;

} // namespace vkcom

namespace std {

template <typename T>
struct hash<vkcom::VectorSegment<T>> {
  uint64_t operator()(const vkcom::VectorSegment<T> &x) const { return x.hash(); }
};

} // namespace std
