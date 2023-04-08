#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <random>

#include "../../youtokentome/cpp/wordpiece.h"

namespace vkcom {

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
    dump_vector("vocab.txt", text.vocab, '\n');
    dump_vector("anwer_encoded.txt", text.anwer_encoded, ' ');
    dump_vector("answer_decoded.txt", text.answer_decoded, ' ');
}

void check(const TestCase &test_case, const std::vector<int> &encoded, const std::vector<std::string> &decoded) {
  if (encoded != test_case.answer_encoded || decoded != test_case.answer_decoded) {
    dump_test_case(test_case);
    throw std::runtime_error("STRESS TEST FAILED, test case dumped");
  }
}

std::string get_random_string(std::mt19937 &rnd, size_t string_length) {
  static constexpr std::string_view kAllChars = "abcdefghijklmnopqrstuvwxyz";
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
    std::uniform_int_distribution<size_t> word_len(0ul, text_len / 2);

    std::unordered_map<std::string, int> vocab_map;
    std::vector<int> answer_encoded;
    std::vector<std::string> answer_decoded;
    text.reserve(parts);

    for (size_t i = 0; i < parts; i++) {
        const size_t vocab_size = vocab_map.size();
        if (i + 1 == parts) {
            size_t leftover = text.capacity() > text.size() ? text.capacity() - text.size() : 0;
            std::string word = get_random_string(rnd, leftover);
            if (vocab_map[word] == 0) {
                vocab_map[word] = static_cast<int>(vocab_size);
            }
            text.append(word);
            answer_encoded.push_back(vocab_map[word]);
            answer_decoded.push_back(std::move(word));
        } else if (text % 10 == 0) {
            std::uniform_int_distribution<size_t> rnd_word(0ul, vocab_size);
            auto it = std::next(vocab_map.begin(), rnd_word(rnd));
            text.append(it->first);
            text.append(' ');
            answer_encoded.push_back(it->second);
            answer_decoded.push_back(it->first);
        } else {
            std::string word = get_random_string(rnd, word_len(rnd));
            if (vocab_map[word] == 0) {
                vocab_map[word] = static_cast<int>(vocab_size);
            }
            text.append(word);
            text.append(' ');
            answer_encoded.push_back(vocab_map[word]);
            answer_decoded.push_back(std::move(word));
        }
    }

    std::vector<std::string> vocab;
    vocab.resize(vocab_map.size());
    for (auto it = vocab_map.begin(); it != vocab_map.end(); it++) {
        vocab[it->second] = it->first;
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

      for (int i = 0; i < 3; i++) {
        TestCase test_case = generate_test_case(text_len, parts, positive);
        std::cout << "running stress, text_len " << test_case.text.size() << ", vocab_size "
                  << test_case.vocab.size() << std::endl;

        Status status;
        std::vector<int> encoded;
        wordpiece::Encoder encoder(test_case.vocab, n_threads);
        status = encoder.encode_as_ids(test_case.text, &encoded);
        if (!status.ok()) {
            dump_test_case(test_case);
            throw std::runtime_error("encode_as_ids failed, test_case dumped");
        }
        std::vector<std::string> decoded;
        status = encoder.encode_as_subwords(test_case.text, test_case.vocab, &decoded);
        if (!status.ok()) {
            dump_test_case(test_case);
            throw std::runtime_error("encode_as_subwords failed, test_case dumped");
        }

        check(test_case, encoded, decoded);
      }
    }
  }
}

void run_small(int n_threads) {
    test_stress(10, 300, 5, 2, 100, n_threads);
    test_stress(10, 300, 5, 2, 100, n_threads);
}

void run_large(int n_threads) {
    test_stress(100'000,
                    1'000'000,
                    400'000,
                    kWordPieceVocabSize,
                    kWordPieceVocabSize,
                    n_threads);
    test_stress(10'000'000,
                10'000'000,
                200'000,
                kWordPieceVocabSize,
                kWordPieceVocabSize,
                n_threads);
}

int main(int argc, char **argv) {
    if (argc == 2 && argv[1] == "small") {
        run_small(1);
    } else if (argc == 2 && argv[1] == "large") {
        run_large(1);
    } else if (argc == 2 && argv[1] == "parallel") {
        run_small(0);
        run_large(0);
    } else {
        assert(false);
    }
}

} // namespace vkcom
