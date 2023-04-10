#include <algorithm>
#include <cassert>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <random>

#include "../../../youtokentome/cpp/wordpiece.h"

using namespace vkcom;

struct TestCase {
    std::string text;
    std::vector<std::string> vocab;
    std::vector<int> answer_encoded;
    std::vector<std::string> answer_decoded;
};

template <typename T>
void dump_vector(const std::string &filename, const std::vector<T> &vec, char delim) {
    std::ofstream fout(filename);
    for (const auto& item : vec) {
        fout << item << delim;
    }
}

void dump_test_case(const TestCase &test_case) {
    {
        std::ofstream fout("stress.txt");
        fout << test_case.text;
    }
    dump_vector("vocab.txt", test_case.vocab, '\n');
    dump_vector("anwer_encoded.txt", test_case.answer_encoded, ' ');
    dump_vector("answer_decoded.txt", test_case.answer_decoded, ' ');
}

void check(const TestCase &test_case, const std::vector<int> &encoded, const std::vector<std::string> &decoded) {
  if (encoded != test_case.answer_encoded || decoded != test_case.answer_decoded) {
    dump_test_case(test_case);
    throw std::runtime_error("STRESS TEST FAILED, test case dumped");
  }
}

std::string get_random_string(std::mt19937 &rnd, size_t string_length) {
  static const std::string kAllChars = "abcdefghijklmnopqrstuvwxyz";
  if (string_length == 0) {
    throw std::runtime_error("string_length cannot be 0");
  }
  std::string result;
  result.reserve(string_length);
  while (string_length > 0) {
    --string_length;
    size_t index = std::uniform_int_distribution<size_t>(0ul, kAllChars.size() - 1)(rnd);
    result.push_back(kAllChars[index]);
  }
  return result;
}

TestCase generate_test_case(size_t text_len, size_t parts) {
    std::mt19937 rnd(17);
    std::string text;
    text.reserve(text_len + parts);
    std::uniform_int_distribution<size_t> word_len(1ul, std::max(2 * text_len / parts, 3ul));

    std::unordered_map<std::string, int> vocab_map;
    std::vector<int> answer_encoded;
    std::vector<std::string> answer_decoded;
    answer_encoded.reserve(parts);
    answer_decoded.reserve(parts);

    for (size_t i = 0; i < parts && text.size() < text.capacity(); i++) {
        const size_t vocab_size = vocab_map.size();
        if (i + 1 == parts) {
            size_t leftover = text.capacity() - text.size();
            std::string word = get_random_string(rnd, leftover);
            if (vocab_map[word] == 0) {
                vocab_map[word] = static_cast<int>(vocab_size) + 1;
            }
            text.append(word);
            answer_encoded.push_back(vocab_map[word] - 1);
            answer_decoded.push_back(std::move(word));
        } else if (i > 0 && i % 10 == 0) {
            std::uniform_int_distribution<size_t> rnd_word(0ul, vocab_size - 1);
            auto it = std::next(vocab_map.begin(), rnd_word(rnd));
            text.append(it->first);
            text.push_back(' ');
            answer_encoded.push_back(it->second - 1);
            answer_decoded.push_back(it->first);
        } else {
            std::string word = get_random_string(rnd, word_len(rnd));
            if (vocab_map[word] == 0) {
                vocab_map[word] = static_cast<int>(vocab_size) + 1;
            }
            text.append(word);
            text.push_back(' ');
            answer_encoded.push_back(vocab_map[word] - 1);
            answer_decoded.push_back(std::move(word));
        }
    }

    std::vector<std::string> vocab;
    vocab.resize(vocab_map.size());
    for (auto it = vocab_map.begin(); it != vocab_map.end(); it++) {
        vocab[it->second - 1] = it->first;
    }
    return TestCase{std::move(text), std::move(vocab), std::move(answer_encoded), std::move(answer_decoded)};
}

void test_stress(size_t text_len_from,
                 size_t text_len_to,
                 size_t text_len_step,
                 size_t parts_from,
                 size_t parts_to,
                 int n_threads) {
  for (size_t text_len = text_len_from; text_len <= text_len_to; text_len += text_len_step) {
    for (size_t parts = std::min(text_len, parts_from); parts <= std::min(text_len, parts_to);
         parts++) {

      const std::string text_filename("stress.txt");
      TestCase test_case = generate_test_case(text_len, parts);
      std::cout << "running stress, text_len " << test_case.text.size() << ' ' << text_len << ", vocab_size "
                << test_case.vocab.size() << std::endl;
      {
        std::ofstream fout(text_filename);
        fout << test_case.text;
      }

      Status status;
      std::vector<int> encoded;
      wordpiece::Encoder encoder(test_case.vocab, n_threads);
      status = encoder.encode_as_ids(text_filename, &encoded);
      if (!status.ok()) {
          dump_test_case(test_case);
          throw std::runtime_error("encode_as_ids failed, test_case dumped: " + status.error_message());
      }
      std::vector<std::string> decoded;
      status = encoder.encode_as_subwords(text_filename, &decoded);
      if (!status.ok()) {
          dump_test_case(test_case);
          throw std::runtime_error("encode_as_subwords failed, test_case dumped: " + status.error_message());
      }
      check(test_case, encoded, decoded);
    }
  }
}

void run_small(int n_threads) {
    test_stress(10, 300, 5, 2, 100, n_threads);
    test_stress(10, 300, 5, 2, 100, n_threads);
}

void run_large(int n_threads) {
    test_stress(100000,
                1000000,
                400000,
                30000,
                30000,
                n_threads);
    test_stress(10000000,
                10000000,
                200000,
                30000,
                30000,
                n_threads);
}

int main(int argc, char **argv) {
    if (argc != 2) {
      assert(false);
    }
    std::string mode = argv[1];
    if (argc == 2 && mode == "small") {
        run_small(1);
    } else if (argc == 2 && mode == "large") {
        run_large(1);
    } else if (argc == 2 && mode == "parallel") {
        run_small(0);
        run_large(0);
    } else {
        assert(false);
    }
}

