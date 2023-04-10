#include "wordpiece.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "utf8.h"

namespace vkcom::wordpiece {

namespace {

const std::string UNK_TOKEN = "[UNK]";
const std::string PAD_TOKEN = "[PAD]";
const std::string BOS_TOKEN = "[BOS]";
const std::string EOS_TOKEN = "[EOS]";

bool is_suffix_vocab(const std::vector<uint32_t> &word) {
  static const uint32_t kSharp = static_cast<uint32_t>('#');
  return word.size() >= 2 && word[0] == kSharp && word[1] == kSharp;
}

bool is_special_token(const std::vector<uint32_t> &word) {
  return word.size() > 2 && word[0] == static_cast<uint32_t>('[')
      && word.back() == static_cast<uint32_t>(']');
}

std::vector<uint32_t> parse_text(const char *text, size_t size, vkcom::ThreadPool &thread_pool) {
  static const size_t kWorkBatch = 5000000;

  if (size < 2 * kWorkBatch) {
    return vkcom::decode_utf8(text, text + size);
  }

  const size_t thread_count = std::min(thread_pool.maxThreads(), size / kWorkBatch);
  const size_t work_batch = size / thread_count + 1;
  std::vector<std::vector<uint32_t>> per_thread_text_utf8(thread_count);
  size_t work_start = 0;
  for (size_t thread_id = 0; thread_id < thread_count && work_start < size; thread_id++) {
    size_t work_end = std::min(size, work_start + work_batch);
    while (work_end < size && !vkcom::check_symbol_start(text[work_end])) {
      ++work_end;
    }
    thread_pool.submit([thread_id, work_start, work_end, text, &per_thread_text_utf8] {
      const char *begin = text + work_start;
      const size_t len = work_end - work_start;
      per_thread_text_utf8[thread_id] = vkcom::decode_utf8(begin, begin + len);
    });
    work_start = work_end;
  }

  thread_pool.waitCompletion();
  size_t text_utf8_size = 0;
  for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
    text_utf8_size += per_thread_text_utf8[thread_id].size();
  }
  std::vector<uint32_t> text_utf8(text_utf8_size);
  text_utf8.resize(text_utf8_size);
  work_start = 0;
  for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
    std::vector<uint32_t> &segment = per_thread_text_utf8[thread_id];
    if (!segment.empty()) {
      std::memcpy(text_utf8.data() + work_start, segment.data(), segment.size() * sizeof(uint32_t));
      work_start += segment.size();
    }
  }

  return text_utf8;
}

std::vector<std::string> read_lines_helper(const std::string &filename) {
  std::ifstream fin(filename);
  return read_all_lines(fin);
}

} // namespace

WordPieceToken::WordPieceToken(const std::string &encoded_word)
 : is_prefix(true), is_special(false), is_malformed(false), word(vkcom::decode_utf8(encoded_word)) {
  if (is_suffix_vocab(word)) {
    is_prefix = false;
    word.erase(word.begin(), word.begin() + 2);
  } else if (is_special_token(word)) {
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

WordPieceVocabulary::WordPieceVocabulary(const std::vector<std::string> &words) {
  tokens.reserve(words.size());
  int token_id = 0;
  max_token_len = 0;
  for (const std::string &word : words) {
    update_special_tokens(word, token_id);
    WordPieceToken token(word);
    max_token_len = std::max(max_token_len, token.word.size());
    tokens.push_back(std::move(token));
    ++token_id;
  }
}

void WordPieceVocabulary::update_special_tokens(const std::string &word, int token_id) {
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

Encoder::Encoder(const std::string &vocab_path, int n_threads)
 : Encoder(read_lines_helper(vocab_path), n_threads) {}

Encoder::Encoder(std::vector<std::string> vocab, int n_threads)
 : vocab_(std::move(vocab)), word_piece_vocab_(vocab_), thread_pool_(n_threads) {
  build_word_maps();
}

Status Encoder::encode_as_ids(const std::string &text_path, std::vector<int> *ids) const {
  try {
    std::string text_str;
    Status status = fast_read_file_utf8(text_path, &text_str);
    if (!status.ok()) {
      return status;
    }
    const std::vector<uint32_t> text = parse_text(text_str.data(), text_str.size(), thread_pool_);
    *ids = encode_parallel(text);
    return Status();
  } catch (const std::exception &ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status Encoder::encode_as_subwords(const std::string &text_path,
                                   std::vector<std::string> *subwords) const {
  try {
    std::string text_str;
    Status status = fast_read_file_utf8(text_path, &text_str);
    if (!status.ok()) {
      return status;
    }
    const std::vector<uint32_t> text = parse_text(text_str.data(), text_str.size(), thread_pool_);
    std::vector<int> ids = encode_parallel(text);
    for (int id : ids) {
      subwords->push_back(vocab_[id]);
    }
    return Status();
  } catch (const std::exception &ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

Status Encoder::decode(const std::vector<int> &ids,
                       std::vector<std::string> *subwords,
                       const std::unordered_set<int> *ignore_ids) const {
  try {
    for (int id : ids) {
      if (!ignore_ids || ignore_ids->count(id) == 0) {
        subwords->push_back(vocab_.at(id));
      }
    }
    return Status();
  } catch (const std::exception &ex) {
    return Status(1, ex.what());
  } catch (...) {
    return Status(1, "Unknown error");
  }
}

bool Encoder::is_word_prefix(const std::vector<uint32_t> &text, size_t index) {
  return index == 0 || vkcom::is_spacing_char(text[index])
      || vkcom::is_spacing_char(text[index - 1]);
}

void Encoder::build_word_maps() {
  for (size_t i = 0; i < word_piece_vocab_.tokens.size(); i++) {
    const auto &token = word_piece_vocab_.tokens[i];
    if (token.is_special || token.is_malformed) {
      continue;
    }
    vkcom::VectorSegmentBuilder<uint32_t> segment(token.word);
    WordMap *word_to_id = token.is_prefix ? &prefix_to_id_ : &suffix_to_id_;
    (*word_to_id)[segment.finish()] = static_cast<int>(i);
  }
}

std::vector<int> Encoder::encode_parallel(const std::vector<uint32_t> &text) const {
  static const size_t kWorkBatch = 1000000;

  if (text.size() < 2 * kWorkBatch) {
    return encode_impl(text, 0, text.size());
  }

  const size_t thread_count = std::min(thread_pool_.maxThreads(), text.size() / kWorkBatch);
  const size_t work_batch = text.size() / thread_count + 1;
  std::vector<std::vector<int>> per_thread_token_ids(thread_count);
  size_t work_begin = 0;
  for (size_t thread_id = 0; thread_id < thread_count && work_begin < text.size(); thread_id++) {
    size_t work_end = std::min(text.size(), work_begin + work_batch);
    while (work_end < text.size() && !vkcom::is_space(text[work_end])) {
      ++work_end;
    }
    thread_pool_.submit([this, thread_id, work_begin, work_end, &per_thread_token_ids, &text] {
      per_thread_token_ids[thread_id] = encode_impl(text, work_begin, work_end);
    });
    work_begin = work_end;
  }

  thread_pool_.waitCompletion();

  size_t token_count = 0;
  for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
    token_count += per_thread_token_ids[thread_id].size();
  }
  std::vector<int> token_ids(token_count);
  work_begin = 0;
  for (size_t thread_id = 0; thread_id < thread_count; thread_id++) {
    std::vector<int> &segment = per_thread_token_ids[thread_id];
    if (!segment.empty()) {
      std::memcpy(token_ids.data() + work_begin, segment.data(), segment.size() * sizeof(int));
      work_begin += segment.size();
    }
  }

  return token_ids;
}

std::vector<int>
Encoder::encode_impl(const std::vector<uint32_t> &text, size_t begin, size_t end) const {
  size_t max_len = std::min(word_piece_vocab_.max_token_len, end - begin);
  if (begin == end) {
    return {};
  }
  if (word_piece_vocab_.tokens.empty()) {
    throw std::runtime_error("abc");
  }
  if (max_len == 0) {
    throw std::runtime_error("her");
  }
  const int unk_token_id = word_piece_vocab_.special_tokens.unk_id;

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
    const WordMap *word_to_id = is_word_prefix(text, begin) ? &prefix_to_id_ : &suffix_to_id_;

    vkcom::VectorSegmentBuilder<uint32_t> segment(segment_begin, segment_end);
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
      while (begin != end && !is_word_prefix(text, begin)) {
        ++begin;
      }
    } else if (begin != end && is_word_prefix(text, begin)) {
      tokens_since_prefix = 0;
    }

    while (begin != end && vkcom::is_space(text[begin])) {
      ++begin;
    }
  }

  return token_ids;
}

} // namespace vkcom::wordpiece
