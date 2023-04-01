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
  static constexpr int kDefaultUnkTokenId = -1;

  std::vector<WordPieceToken> tokens;
  int unk_token_id = kDefaultUnkTokenId;
};

WordPieceVocabulary readVocabFromFile(const std::string &file) {
  WordPieceVocabulary vocab_utf8;
  std::ifstream fin(file);
  std::string word;
  int token_id = 0;
  while (std::getline(fin, word)) {
    if (word == kUnkTokenIdStr) {
      vocab_utf8.unk_token_id = token_id;
    }
    WordPieceToken token(word);
    vocab_utf8.tokens.push_back(std::move(token));
    ++token_id;
  }
  return vocab_utf8;
}

vkcom::ThreadPool &globalThreadPool(size_t n_threads) {
  static vkcom::ThreadPool thread_pool(n_threads);
  return thread_pool;
}

std::vector<int> encodeWordPieceImpl(const std::vector<uint32_t> &text,
                                     const WordPieceVocabulary &vocab) {
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

  const auto worker = [&, unk_token_id = vocab.unk_token_id](size_t begin, size_t end) {
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
    const size_t thread_count = std::min(globalThreadPool().maxThreads(), text.size() / kWorkBatch);
    const size_t work_batch = text.size() / thread_count + 1;
    std::vector<std::vector<int>> per_thread_token_ids(thread_count);
    size_t work_begin = 0;
    for (size_t thread_id = 0; thread_id < thread_count && work_begin < text.size(); thread_id++) {
      size_t work_end = std::min(text.size(), work_begin + work_batch);
      while (work_end < text.size() && !vkcom::is_space(text[work_end])) {
        ++work_end;
      }
      globalThreadPool().submit([thread_id, work_begin, work_end, &per_thread_token_ids, &worker] {
        per_thread_token_ids[thread_id] = worker(work_begin, work_end);
      });
      work_begin = work_end;
    }

    globalThreadPool().waitCompletion();

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

std::vector<int> encodeWordPiece(const char *text, size_t size, const WordPieceVocabulary &vocab) {
  if (size == 0) {
    return {};
  }
  const std::vector<uint32_t> text_utf8 = utils::parseText(text, size, globalThreadPool());
  return encodeWordPieceImpl(text_utf8, vocab);
}

} // namespace

namespace vkcom::wordpiece {

/*std::vector<int> encode_wordpiece(const std::string& input_path, const std::string& vocab_path) {
  const WordPieceVocabulary vocab_utf8 = readVocabFromFile(vocab_file);
  // TODO: use mapped file
  std::string text = read_file(input_path);
  return encodeWordPiece(text.data(), text.size(), vocab_utf8);
}

Status encode_wordpiece_cli(const std::string& input_path, const std::string& vocab_path) {
  try {
    std::vector<int> ids = encode_wordpiece(input_path, vocab_path);
    write_to_stdout(ids, true);
    return Status();
  } catch (const std::exception& ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}*/

} // namespace vkcom::wordpiece