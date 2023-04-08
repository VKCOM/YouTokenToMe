#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "third_party/thread_pool/thread_pool.h"
#include "utils.h"

namespace vkcom::wordpiece {

struct WordPieceToken {
  explicit WordPieceToken(const std::string &encoded_word);

  bool is_prefix;
  bool is_special;
  bool is_malformed;
  std::vector<uint32_t> word;
};

struct WordPieceVocabulary {
  explicit WordPieceVocabulary(const std::vector<std::string> &words);

  std::vector<WordPieceToken> tokens;
  vkcom::SpecialTokens special_tokens;
  size_t max_token_len = 0;

 private:
  void update_special_tokens(const std::string &word, int token_id);
};

class Encoder {
 public:
  explicit Encoder(const std::string &vocab_path, int n_threads);

  explicit Encoder(std::vector<std::string> vocab, int n_threads);

  Status encode_as_ids(const std::string &text_path, std::vector<int> *ids) const;

  Status encode_as_subwords(const std::string &text_path, std::vector<std::string> *subwords) const;

  Status decode(const std::vector<int> &ids,
                std::vector<std::string> *subwords,
                const std::unordered_set<int> *ignore_ids) const;

  Status id_to_subword(int id, std::string *subword) const;

  int subword_to_id(const std::string &token) const;

 private:
  static const uint64_t kReadBatchLimit = 10 * 1024 * 1024;

  static bool is_word_prefix(const std::vector<uint32_t> &text, size_t index);

  void build_word_maps();

  std::vector<int> encode_parallel(const std::vector<uint32_t> &text);
  std::vector<int> encode_impl(const std::vector<uint32_t> &text, size_t begin, size_t end) const;

  std::vector<std::string> vocab_;
  WordPieceVocabulary word_piece_vocab_;

  // TODO: flat_hash_map ?
  using WordMap = std::unordered_map<vkcom::WordPieceVectorSegment, int>;
  WordMap prefix_to_id_; // no ## in word prefix
  WordMap suffix_to_id_; // ## in word prefix

  mutable ThreadPool thread_pool_;
};

} // namespace vkcom::wordpiece
