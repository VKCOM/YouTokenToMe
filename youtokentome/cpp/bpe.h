#pragma once

#include <map>
#include <string>
#include <unordered_set>
#include "third_party/flat_hash_map/flat_hash_map/flat_hash_map.h"

#include "utils.h"

namespace vkcom {

Status train_bpe(const std::string &input_path, const std::string &model_path,
                 int vocab_size, BpeConfig config);

Status learn_bpe_from_string(std::string &text_utf8,
                             int n_tokens,
                             const std::string &output_file,
                             BpeConfig bpe_config,
                             BPEState *bpe_state);

flat_hash_map<uint32_t, uint32_t>
compute_alphabet_helper(const flat_hash_map<uint32_t, uint64_t> &char_cnt,
                        uint64_t data_len,
                        flat_hash_set<uint32_t> &removed_chars,
                        const BpeConfig &bpe_config);

class BaseEncoder {
 public:
  BPEState bpe_state;
  flat_hash_map<uint32_t, uint32_t> id2char;
  flat_hash_map<uint32_t, std::vector<uint32_t>> recipe;
  flat_hash_map<std::string, uint32_t> reversed_recipe;
  flat_hash_map<uint64_t, int> rule2id;
  int n_threads;

  explicit BaseEncoder(BPEState bpe_state, int _n_threads);

  explicit BaseEncoder(const std::string &model_path, int n_threads, Status *ret_status);

  void fill_from_state();

  Status encode_as_ids(
      const std::vector<std::string> &sentences, std::vector<std::vector<int>> *ids, bool bos = false,
      bool eos = false, bool reverse = false, double dropout_prob=0) const;

  Status encode_as_subwords(
      const std::vector<std::string> &sentences,
      std::vector<std::vector<std::string>> *subwords,
      bool bos = false,
      bool eos = false, bool reverse = false, double dropout_prob=0) const;

  Status id_to_subword(int id, std::string *subword, bool replace_space = false) const;

  int subword_to_id(const std::string &token) const;

  Status decode(const std::vector<std::vector<int>> &ids,
                std::vector<std::string> *sentences,
                const std::unordered_set<int> *ignore_ids) const;

  Status decode(const std::vector<int> &ids, std::string *sentence, const std::unordered_set<int> *ignore_ids) const;

  Status decode(const std::vector<std::string> &ids,
                std::vector<std::string> *sentences,
                const std::unordered_set<int> *ignore_ids) const;

  int vocab_size() const;

  std::vector<std::string> vocabulary() const;

  Status encode_cli(const std::string &output_type, bool stream, bool bos = false,
                    bool eos = false, bool reverse = false, double dropout_prob = 0) const;

  Status decode_cli(const std::unordered_set<int> *ignore_ids) const;

  void vocab_cli(bool verbose) const;

 private:
  DecodeResult encode_sentence(const std::string &sentence_utf8,
                               const EncodingConfig &encoding_config,
                               OutputType output_type) const;

  Status encode_parallel(
      const std::vector<std::string> &sentences,
      const EncodingConfig &encoding_config, OutputType output_type,
      std::vector<DecodeResult> *decoder_results
  ) const;
};

} // namespace vkcom
