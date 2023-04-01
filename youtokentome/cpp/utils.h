#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "third_party/flat_hash_map/flat_hash_map.h"

namespace vkcom {

const std::string UNK_TOKEN = "<UNK>";
const std::string PAD_TOKEN = "<PAD>";
const std::string BOS_TOKEN = "<BOS>";
const std::string EOS_TOKEN = "<EOS>";

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

struct BPE_Rule {
  // x + y -> z
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};

  BPE_Rule() = default;

  BPE_Rule(uint32_t x, uint32_t y, uint32_t z);

  bool operator==(const BPE_Rule &other) const;
};

struct BpeConfig {
  double character_coverage = 1;
  int n_threads = 0;
  SpecialTokens special_tokens;

  BpeConfig() = default;

  BpeConfig(double character_coverage, int n_threads,
            const SpecialTokens &special_tokens);
};

struct BPEState {
  flat_hash_map<uint32_t, uint32_t> char2id;
  std::vector<BPE_Rule> rules;
  SpecialTokens special_tokens;

  void dump(const std::string &file_name);

  Status load(const std::string &file_name);
};

struct EncodingConfig {
  bool bos;
  bool eos;
  bool reverse;
  double dropout_prob;
};

std::vector<std::string> read_lines_from_stdin(uint64_t batch_limit, uint64_t *processed);

std::string read_file(const std::string& path);

template<typename T>
void write_to_stdout(const std::vector<T> &items, bool flush) {
  for (const auto &item : items) {
    std::cout << item << " ";
  }
  std::cout << "\n";
  if (flush) {
    std::cout << std::flush;
  }
}

template<typename T>
void write_to_stdout(const std::vector<std::vector<T>> &sentences, bool flush) {
  for (const auto &sentence : sentences) {
    write_to_stdout(sentence, false);
  }
  if (flush) {
    std::cout << std::flush;
  }
}

class VectorSegmentBuilder;

struct VectorSegment {
  private:
    friend class VectorSegmentBuilder;

    const uint32_t *begin_;
    const uint32_t *end_;
    const uint64_t hash_;

    VectorSegment(const uint32_t *begin, const uint32_t *end, uint64_t hash)
        : begin_(begin), end_(end), hash_(hash) {}

  public:
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

class VectorSegmentBuilder {
  private:
    constexpr static uint64_t MOD = 2032191299;
    constexpr static uint64_t P = 726328703;

    const uint32_t *begin_;
    const uint32_t *end_;
    std::vector<uint64_t> prefix_hash_;

  public:
    VectorSegmentBuilder(const std::vector<uint32_t> &segment)
        : VectorSegmentBuilder(segment.data(), segment.data() + segment.size()) {}

    VectorSegmentBuilder(const uint32_t *begin, const uint32_t *end) : begin_(begin), end_(end) {
        uint64_t hash = 0;
        prefix_hash_.reserve(static_cast<size_t>(end - begin));
        for (const uint32_t *it = begin_; it != end_; it++) {
            hash = (hash * P + *it) % MOD;
            prefix_hash_.push_back(hash);
        }
    }

    VectorSegment finish() const { return VectorSegment(begin_, end_, hash()); }

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

} // namespace vkcom

namespace std {

template <>
struct hash<vkcom::VectorSegment> {
    uint64_t operator()(const vkcom::VectorSegment &x) const { return x.hash(); }
};

} // namespace std