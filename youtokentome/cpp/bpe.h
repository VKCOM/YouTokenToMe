#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include "third_party/flat_hash_map.h"

#include "utils.h"

namespace vkcom {

const std::string UNK_TOKEN = "<UNK>";
const std::string PAD_TOKEN = "<PAD>";
const std::string BOS_TOKEN = "<BOS>";
const std::string EOS_TOKEN = "<EOS>";

enum OutputType { ID, SUBWORD };

void train_bpe(const std::string& input_path, const std::string& model_path,
               int vocab_size, BpeConfig config);

void print_vocab(const std::string& model_path, bool verbose);

class BaseEncoder {
 public:
  BPEState bpe_state;
  ska::flat_hash_map<uint32_t, uint32_t> id2char;
  ska::flat_hash_map<uint32_t, std::vector<uint32_t>> recipe;
  ska::flat_hash_map<std::string, uint32_t> reversed_recipe;
  ska::flat_hash_map<uint64_t, int> rule2id;
  int n_threads;

  explicit BaseEncoder(BPEState bpe_state, int _n_threads);

  explicit BaseEncoder(const std::string& model_path, int n_threads);

  void fill_from_state();

  std::vector<std::vector<int>> encode_as_ids(
      const std::vector<std::string>& sentences, bool bos = false,
      bool eos = false, bool reverse = false) const;

  std::vector<std::vector<std::string>> encode_as_subwords(
      const std::vector<std::string>& sentences, bool bos = false,
      bool eos = false, bool reverse = false) const;

  std::string id_to_subword(int id, bool replace_space = false) const;

  int subword_to_id(const std::string& token) const;

  std::vector<std::string> decode(const std::vector<std::vector<int>>& ids) const;

  std::string decode(const std::vector<int>& ids) const;

  std::vector<std::string> decode(const std::vector<std::string>& ids) const;

  int vocab_size() const;

  std::vector<std::string> vocabulary() const;

  void encode_cli(const std::string& output_type, bool stream, bool bos = false,
                  bool eos = false, bool reverse = false) const;

  void decode_cli() const;

  void vocab_cli(bool verbose) const;

 private:
  DecodeResult encode_sentence(const std::string& sentence_utf8,
                               const EncodingConfig& encoding_config,
                               OutputType output_type) const;

  std::vector<DecodeResult> encode_parallel(
      const std::vector<std::string>& sentences,
      const EncodingConfig& encoding_config, OutputType output_type) const;
};

}  // namespace vkcom
