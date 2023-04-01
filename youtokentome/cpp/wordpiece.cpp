#include "wordpiece.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/flat_hash_map/flat_hash_map.h"
#include "third_party/thread_pool/thread_pool.hpp"
#include "utf8.h"

namespace {

const std::string UNK_TOKEN = "[UNK]";
const std::string PAD_TOKEN = "[PAD]";
const std::string BOS_TOKEN = "[BOS]";
const std::string EOS_TOKEN = "[EOS]";

struct WordPieceToken {
  explicit WordPieceToken(const std::string &encoded_word)
   : is_prefix(true), is_special(false), is_malformed(false),
     word(vkcom::decode_utf8(encoded_word)) {
    if (isSuffixVocab(word)) {
      is_prefix = false;
      word.erase(word.begin(), word.begin() + 2);
    } else if (isSpecialToken(word)) {
      is_special = true;
    }

    bool all_punctuation = true;
    for (uint32_t code_point : word) {
      if (code_point == vkcom::INVALID_UNICODE) {
        is_malformed = true;
      }
      if (!vkcom::is_punctuation(code_point) && !vkcom::is_space(code_point)) {
        all_punctuation = false;
      }
    }
    if (word.empty()) {
      throw std::runtime_error("Vocab word is empty");
    }
    if (is_malformed || (all_punctuation && word.size() > 1)) {
      is_malformed = true;
      std::cerr << "Vocab word is malformed: " << encoded_word << std::endl;
    }
  }

  bool is_prefix;
  bool is_special;
  bool is_malformed;
  std::vector<uint32_t> word;
};

struct WordPieceVocabulary {
  explicit WordPieceVocabulary(const std::vector<std::string>& words) {
    tokens.reserve(words.size());
    int token_id = 0;
    for (const std::string& word : words) {
      update_special_tokens(word, token_id);
      WordPieceToken token(word);
      tokens.push_back(std::move(token));
      ++token_id;
    }
  }

  explicit WordPieceVocabulary(const std::string &file) {
    WordPieceVocabulary vocab_utf8;
    std::ifstream fin(file);
    std::string word;
    int token_id = 0;
    while (std::getline(fin, word)) {
      update_special_tokens(word, token_id);
      WordPieceToken token(word);
      tokens.push_back(std::move(token));
      ++token_id;
    }
  }

  std::vector<WordPieceToken> tokens;
  SpecialTokens special_tokens;

private:
  void update_special_tokens(const std::string& word, int token_id) {
    if (word == UNK_TOKEN) {
      special_tokens.unk_id = token_id;
    } else if (word == PAD_TOKEN) {
      special_tokens.pad_id = token_id;
    } else if (word == BOS_TOKEN) {
      special_tokens.bos_id = token_id;
    } else if (word == EOS_TOKEN) {
      special_tokens.eos_id = token_id;
    }
  }
};

std::vector<int> encode_word_piece_impl(const std::vector<uint32_t> &text,
                                        const WordPieceVocabulary &vocab,
                                        vkcom::ThreadPool& thread_pool) {
  using WordMap = std::unordered_map<vkcom::VectorSegment, int>;
  WordMap prefix_to_id; // no ## in word prefix
  WordMap suffix_to_id; // ## in word prefix

  size_t max_len = 0;
  for (size_t i = 0; i < vocab.tokens.size(); i++) {
    const auto &token = vocab.tokens[i];
    if (token.is_special || token.is_malformed) {
      continue;
    }
    max_len = std::max(max_len, token.word.size());
    vkcom::VectorSegmentBuilder segment(token.word);
    WordMap *word_to_id = token.is_prefix ? &prefix_to_id : &suffix_to_id;
    (*word_to_id)[segment.finish()] = static_cast<int>(i);
  }
  max_len = std::min(max_len, text.size());

  const auto is_word_prefix = [&text](size_t index) {
    return index == 0 || vkcom::is_spacing_char(text[index])
        || vkcom::is_spacing_char(text[index - 1]);
  };

  const auto worker = [&, unk_token_id = vocab.special_tokens.unk_id](size_t begin, size_t end) {
    std::vector<int> token_ids;
    token_ids.reserve((end - begin) / max_len + 1);

    while (begin != end && vkcom::is_space(text[begin])) {
      ++begin;
    }

    size_t tokens_since_prefix = 0;

    while (begin != end) {
      size_t word_len = 1;
      if (!vkcom::is_punctuation(text[begin])) {
        while (word_len < std::min(max_len, end - begin)
               && !vkcom::is_spacing_char(text[begin + word_len])) {
          ++word_len;
        }
      }

      const uint32_t *segment_begin = text.data() + static_cast<int64_t>(begin);
      const uint32_t *segment_end = segment_begin + static_cast<int64_t>(word_len);
      const WordMap *word_to_id = is_word_prefix(begin) ? &prefix_to_id : &suffix_to_id;

      vkcom::VectorSegmentBuilder segment(segment_begin, segment_end);
      while (!segment.empty()) {
        auto it = word_to_id->find(segment.finish());
        if (it != word_to_id->end()) {
          ++tokens_since_prefix;
          token_ids.push_back(it->second);
          begin += segment.size();
          break;
        } else {
          segment.pop_back();
        }
      }

      if (segment.empty()) {
        while (tokens_since_prefix > 0) {
          token_ids.pop_back();
          --tokens_since_prefix;
        }
        token_ids.push_back(unk_token_id);
        begin += word_len;
        while (begin != end && !is_word_prefix(begin)) {
          ++begin;
        }
      } else if (begin != end && is_word_prefix(begin)) {
        tokens_since_prefix = 0;
      }

      while (begin != end && vkcom::is_space(text[begin])) {
        ++begin;
      }
    }

    return token_ids;
  };

  static constexpr size_t kWorkBatch = 1'000'000;
  std::vector<int> token_ids;
  if (text.size() < 2 * kWorkBatch) {
    token_ids = worker(0, text.size());
  } else {
    const size_t thread_count = std::min(thread_pool.maxThreads(), text.size() / kWorkBatch);
    const size_t work_batch = text.size() / thread_count + 1;
    std::vector<std::vector<int>> per_thread_token_ids(thread_count);
    size_t work_begin = 0;
    for (size_t thread_id = 0; thread_id < thread_count && work_begin < text.size(); thread_id++) {
      size_t work_end = std::min(text.size(), work_begin + work_batch);
      while (work_end < text.size() && !vkcom::is_space(text[work_end])) {
        ++work_end;
      }
      thread_pool.submit([thread_id, work_begin, work_end, &per_thread_token_ids, &worker] {
        per_thread_token_ids[thread_id] = worker(work_begin, work_end);
      });
      work_begin = work_end;
    }

    thread_pool.waitCompletion();

    size_t token_count = 0;
    for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
      token_count += per_thread_token_ids[thread_id].size();
    }
    token_ids.resize(token_count);
    work_begin = 0;
    for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
      std::vector<int> &segment = per_thread_token_ids[thread_id];
      if (!segment.empty()) {
        std::memcpy(token_ids.data() + work_begin, segment.data(), segment.size() * sizeof(int));
        work_begin += segment.size();
      }
    }
  }

  return token_ids;
}

std::vector<int> encode_word_piece(const char *text, size_t size, const WordPieceVocabulary &vocab) {
  if (size == 0) {
    return {};
  }
  vkcom::ThreadPool thread_pool(0);
  const std::vector<uint32_t> text_utf8 = utils::parseText(text, size, thread_pool);
  return encode_word_piece_impl(text_utf8, vocab, thread_pool);
}

} // namespace

namespace vkcom::wordpiece {

Status encode_as_ids(const std::string &text_path,
                     const std::string& vocab_path, std::vector<int> *ids) {
  const uint64_t batch_limit = 10 * 1024 * 1024;
  try {
    std::string text;
    Status status = fast_read_file_utf8(text_path, &text);
    if (!status.ok()) {
      return status;
    }
    uint64_t processed = 0;
    std::vector<std::string> vocab = read_lines_from_stdin(batch_limit, &processed);
    return encode_as_ids(text, vocab, ids);
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status encode_as_ids(const std::string &text,
                     const std::vector<std::string>& vocab, std::vector<int> *ids) {
  try {
    WordPieceVocabulary word_piece_vocab(vocab);
    *ids = encode_word_piece(text.data(), text.size(), word_piece_vocab);
    return Status();
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status encode_as_subwords(const std::string &text_path,
                          const std::string& vocab_path,
                          std::vector<std::string> *subwords) {
  try {
    std::string text;
    Status status = fast_read_file_utf8(text_path, &text);
    if (!status.ok()) {
      return status;
    }
    uint64_t processed = 0;
    std::vector<std::string> vocab = read_lines_from_stdin(batch_limit, &processed);
    return encode_as_subwords(text, vocab, subwords);
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status encode_as_subwords(const std::string &text,
                          const std::vector<std::string>& vocab,
                          std::vector<std::string> *subwords) {
  try {
    WordPieceVocabulary word_piece_vocab(vocab);
    std::vector<int> ids = encode_word_piece(text.data(), text.size(), word_piece_vocab);
    for (int id : ids) {
      subwords->push_back(vocab[id]);
    }
    return Status();
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status decode(const std::vector<int>& ids,
              const std::vector<std::string>& vocab,
              std::vector<std::string> *subwords,
              const std::unordered_set<int> *ignore_ids) {
  try {
    for (int id : ids) {
      if (!ignore_ids || ignore_ids->count(id) == 0) {
        subwords->push_back(vocab[id]);
      }
    }
    return Status();
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

} // namespace vkcom::wordpiece