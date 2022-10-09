#pragma once

#include "third_party/flat_hash_map.h"
#include <iostream>
#include <string>
#include <vector>

namespace vkcom {
const uint32_t SPACE_TOKEN = 9601;

struct BPE_Rule {
  // x + y -> z
  uint32_t x;
  uint32_t y;
  uint32_t z;
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

struct BpeConfig {
  double character_coverage = 1;
  int n_threads = 0;
  SpecialTokens special_tokens;

  BpeConfig() = default;

  BpeConfig(double character_coverage, int n_threads, const SpecialTokens &special_tokens);
};

struct Status {
  int code{0};
  std::string message;
  Status() = default;
  Status(int code, std::string message);

  const std::string &error_message() const;
  bool ok() const;
};

struct BPEState {
  flat_hash_map<uint32_t, uint32_t> char2id;
  std::vector<BPE_Rule> rules;
  SpecialTokens special_tokens;

  void dump(const std::string &file_name);

  Status load(const std::string &file_name);
};

struct MergeCandidate {
  uint64_t count{0};
  uint32_t left_token{0};
  uint32_t right_token{0};

  MergeCandidate() = default;

  MergeCandidate(uint64_t count, uint32_t left_token, uint32_t right_token)
   : count(count), left_token(left_token), right_token(right_token) {}

  bool operator<(const MergeCandidate &other) const {
    if (count != other.count) {
      return count < other.count;
    }
    uint32_t this_min = std::min(left_token, right_token);
    uint32_t this_max = std::max(left_token, right_token);

    uint32_t other_min = std::min(other.left_token, other.right_token);
    uint32_t other_max = std::max(other.left_token, other.right_token);
    if (this_max != other_max) {
      return this_max > other_max;
    }
    if (this_min != other_min) {
      return this_min > other_min;
    }
    return left_token < other.left_token;
  }
};

struct DecodeResult {
  std::vector<int> ids;
  std::vector<std::string> pieces;
};

struct EncodingConfig {
  bool bos;
  bool eos;
  bool reverse;
  double dropout_prob;
};

bool is_space(uint32_t ch);

std::vector<std::string> read_lines_from_stdin(uint64_t batch_limit, uint64_t *processed);

template <typename T>
void write_to_stdout(const std::vector<std::vector<T>> &sentences, bool flush) {
  for (const auto &sentence : sentences) {
    for (const auto &token : sentence) {
      std::cout << token << " ";
    }
    std::cout << "\n";
  }
  if (flush) {
    std::cout << std::flush;
  }
}

} // namespace vkcom
