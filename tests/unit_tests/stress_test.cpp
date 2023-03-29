#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include "stress_test.h"

#include "../../youtokentome/cpp/utils.h"
#include "../../youtokentome/cpp/bpe.h"
#include "../../youtokentome/cpp/utf8.h"

namespace vkcom {

using namespace std;

// random number from the range [l, r)
long long uniform_dist_int(mt19937& rnd, long long l, long long r) {
  uint64_t range = r - l;
  const uint64_t generator_range = rnd.max() - rnd.min();
  assert(generator_range >= range);
  if (range == 1)
    return l;

  const uint64_t reject = generator_range % range;
  if (reject + 1 == range)
    return (rnd() - rnd.min()) % range + l;

  uint64_t n;
  do {
    n = rnd() - rnd.min();
  } while (n <= reject);
  return (n % range) + l;
}

double uniform_dist_double(mt19937& rnd, double l, double r) {
    const uint64_t generator_range = rnd.max() - rnd.min();
    return static_cast<double>(rnd() - rnd.min()) / generator_range * (r - l) + l;
};

flat_hash_map<uint32_t, uint32_t> compute_alphabet(
    const std::vector<uint32_t> &data,
    flat_hash_set<uint32_t> &removed_chars,
    const BpeConfig &bpe_config) {
  flat_hash_map<uint32_t, uint64_t> char_cnt;
  for (auto ch : data) {
    if (!is_space(ch)) {
      char_cnt[ch]++;
    }
  }
  assert(!char_cnt.empty());
  return compute_alphabet_helper(char_cnt, data.size(), removed_chars,
                                 bpe_config);
}

void remove_rare_chars(std::vector<uint32_t> &data,
                       const flat_hash_set<uint32_t> &removed_chars) {
  if (removed_chars.empty()) {
    return;
  }
  auto it_first_rare_char =
      std::remove_if(data.begin(), data.end(),
                     [&](uint32_t c) { return removed_chars.count(c) != 0; });
  data.erase(it_first_rare_char, data.end());
}

BPEState learn_bpe_slow(const string &text_utf8, int n_token, string, BpeConfig bpe_config) {
  auto row_data = decode_utf8(text_utf8.data(), text_utf8.data() + text_utf8.size());
  vector<vector<uint32_t>> splited_text;
  for (auto &ch: row_data) {
    if (is_space(ch)) {
      ch = SPACE_TOKEN;
    }
  }
  for (; !row_data.empty() && is_space(row_data.back()); row_data.pop_back());
  flat_hash_set<uint32_t> removed_chars;
  auto char2id = compute_alphabet(row_data, removed_chars, bpe_config);
  remove_rare_chars(row_data, removed_chars);
  flat_hash_map<uint32_t, uint32_t> id2char;
  for (auto x: char2id) {
    id2char[x.second] = x.first;
  }
  int used_ids = bpe_config.special_tokens.n_special_tokens() + char2id.size();

  for (int i = 0; i < (int) row_data.size();) {
    for (; i < (int) row_data.size() && is_space(row_data[i]); i++);
    if (i == (int) row_data.size()) {
      break;
    }
    splited_text.emplace_back();
    splited_text.back().push_back(SPACE_TOKEN);
    for (; i < (int) row_data.size() && !is_space(row_data[i]); i++) {
      if (char2id.count(row_data[i])) {
        splited_text.back().push_back(row_data[i]);
      }
    }
  }
  vector<vector<int>> coded;

  for (const auto &v: splited_text) {
    coded.emplace_back();
    for (auto ch: v) {
      coded.back().push_back(char2id[ch]);
    }
  }

  map<int, vector<int>> recipe;
  map<int, string> recipe_s;
  for (int i = 2; i < used_ids; i++) {
    recipe[i] = {i};
    recipe_s[i] = encode_utf8({id2char[i]});
  }

  auto get_recipe = [&](int x, int y) {
    assert(recipe.count(x));
    assert(recipe.count(y));
    vector<int> target_recipe;
    for (auto token_id: recipe[x]) target_recipe.push_back(token_id);
    for (auto token_id: recipe[y]) target_recipe.push_back(token_id);
    return target_recipe;
  };

  struct Candidate {
    uint32_t x, y;
    int cnt;
    bool operator<(const Candidate &other) const {
      if (cnt != other.cnt) {
        return cnt < other.cnt;
      }
      auto this_mn = min(x, y);
      auto this_mx = max(x, y);

      auto other_mn = min(other.x, other.y);
      auto other_mx = max(other.x, other.y);

      if (this_mx != other_mx) {
        return this_mx > other_mx;
      }
      if (this_mn != other_mn) {
        return this_mn > other_mn;
      }
      return x < other.x;
    }
  };

  vector<BPE_Rule> rules;

  for (; used_ids < n_token;) {
    map<pair<int, int>, int> local_cnt;

    for (const auto &v: coded) {
      for (int i = 0; i < (int) v.size() - 1; i++) {
        local_cnt[{v[i], v[i + 1]}]++;
        if (v[i] == v[i + 1] && i + 2 < (int) v.size() && v[i] == v[i + 2]) {
          i++;
        }
      }
    }

    Candidate best = {0, 0, -1};

    for (auto cand: local_cnt) {
      uint32_t x = cand.first.first;
      uint32_t y = cand.first.second;
      Candidate cur = {x, y, cand.second};
      if (best < cur) {
        best = cur;
      }
    }

    if (best.cnt == -1) {
      break;
    }
    uint32_t z = used_ids++;
    rules.push_back({best.x, best.y, z});

    recipe[z] = get_recipe(best.x, best.y);
    recipe_s[z] = recipe_s[best.x] + recipe_s[best.y];

    for (auto &v: coded) {
      for (int i = 0; i < (int) v.size() - 1; i++) {
        if (v[i] == static_cast<long long>(best.x) && v[i + 1] == static_cast<long long>(best.y)) {
          v[i] = z;
          v.erase(v.begin() + i + 1);
        }
      }
    }
  }

  BPEState state = {char2id, rules, bpe_config.special_tokens};
  return state;
}

DecodeResult decode_slow(const string &text_utf8, const BaseEncoder &bpe_applyer) {

  const auto &char2id = bpe_applyer.bpe_state.char2id;
  const auto &id2char = bpe_applyer.id2char;
  const auto &rules = bpe_applyer.bpe_state.rules;
  const auto &recipe = bpe_applyer.recipe;

  auto text = decode_utf8(text_utf8.data(), text_utf8.data() + text_utf8.size());
  for (auto &ch: text) {
    if (is_space(ch)) {
      ch = SPACE_TOKEN;
    }
  }

  for (; !text.empty() && text.back() == SPACE_TOKEN; text.pop_back());

  struct Node {
    uint32_t val;
    string new_chars;
  };

  vector<vector<Node>> words;
  for (int i = 0; i < (int) text.size();) {
    for (; i < (int) text.size() && is_space(text[i]); i++);
    if (i == (int) text.size()) {
      break;
    }

    words.emplace_back();
    words.back().push_back({char2id.at(SPACE_TOKEN), {}});
    for (; i < (int) text.size() && !is_space(text[i]);) {

      if (char2id.count(text[i]) == 0) {
        int cur = i;
        for (; i < (int) text.size() && !is_space(text[i]) && char2id.count(text[i]) == 0; i++);
        words.back().push_back({static_cast<uint32_t>(bpe_applyer.bpe_state.special_tokens.unk_id),
                                encode_utf8({text.begin() + cur, text.begin() + i})});
      } else {
        words.back().push_back({char2id.at(text[i]), {}});
        i++;
      }
    }
  }

  for (auto rule: rules) {
    for (auto &v: words) {
      for (int i = 0; i + 1 < (int) v.size(); i++) {
        if (v[i].val == rule.x && v[i + 1].val == rule.y) {
          v[i].val = rule.z;
          v.erase(v.begin() + i + 1);
        }
      }
    }
  }

  vector<int> ids;
  vector<string> pieces;
  for (auto &v: words) {
    for (const auto &u: v) {
      ids.push_back(u.val);
      if (static_cast<long long>(u.val) == bpe_applyer.bpe_state.special_tokens.unk_id) {
        pieces.push_back(u.new_chars);
      } else {
        auto recipe_u = recipe.at(u.val);
        vector<uint32_t> recipe_u_utf8;
        for (auto ch: recipe_u) {
          assert(id2char.count(ch));
          recipe_u_utf8.push_back(id2char.at(ch));
        }
        pieces.push_back(encode_utf8(recipe_u_utf8));
      }
    }
  }

  return {ids, pieces};
}

string generate_text(int n_limit, bool flag_train, mt19937& rnd) {
  string sigma = flag_train ? "abc " : "abcd ";
  vector<uint32_t> a;
  int n = uniform_dist_int(rnd, 1, 1001);
  n = min(n, n_limit);
  string row_data;
  row_data.push_back(sigma[0]);

  auto add_char = [&](char ch) {
    row_data.push_back(ch);
  };

  for (; (int) row_data.size() < n;) {
    if (uniform_dist_int(rnd, 0, 2)) {
      add_char(sigma[uniform_dist_int(rnd, 0, sigma.size())]);
    } else {
      int l = uniform_dist_int(rnd, 2, 7);
      int seg = uniform_dist_int(rnd, 1, 5);
      vector<uint32_t> tmp;
      for (int i = 0; i < seg; i++) {
        add_char(sigma[uniform_dist_int(rnd, 0, sigma.size())]);
      }
      for (int i = 0; i < l; i++) {
        for (auto ch: tmp) {
          add_char(ch);
        }
      }
    }
  }
  if ((int) row_data.size() > n) {
    row_data.resize(n);
  }
  for (; !row_data.empty() && is_space(row_data.back()); row_data.pop_back());
  for (; (int) row_data.size() < n;) {
    row_data.push_back(sigma[0]);
  }
  assert(static_cast<long long>(row_data.size()) >= n);
  return row_data;

}

void manual_test() {
  Status status;
  string trn_data = "baba baaab";
  string inf_data = "d d";
  int n_tokens = 2 + 2 + 5;

  auto trn_data_copy = trn_data;
  SpecialTokens special_tokens_config = {0, 1, 2, 3};
  BpeConfig bpe_config = {1.0, 1, special_tokens_config};

  BPEState model_fast;
  status = learn_bpe_from_string(trn_data_copy, n_tokens, "remove_it.txt", bpe_config, &model_fast);
  assert(status.ok());
  auto model_slow = learn_bpe_slow(trn_data, n_tokens, "remove_it.txt", bpe_config);
  assert(model_fast.rules == model_slow.rules);
  assert(model_fast.char2id == model_slow.char2id);

  BaseEncoder applyer(model_fast, 1);
  vector<vector<int>> ids_tmp;
  status  = applyer.encode_as_ids({inf_data}, &ids_tmp);
  assert(status.ok());
  auto ids = ids_tmp[0];
  auto result_slow = decode_slow(inf_data, applyer);
  assert(ids == result_slow.ids);
}

vector<uint32_t> to_no_space_tokens(string raw_string) {
  auto tokens = decode_utf8(raw_string.data(), raw_string.data() + raw_string.size());
  int cur = 0;
  for (auto ch: tokens) {
    if (!is_space(ch)) {
      tokens[cur++] = ch;
    }
  }
  tokens.resize(cur);
  return tokens;
}

void parallel_test(int n_iter, int n_threads) {
  mt19937 rnd;
  Status status;
  for (int i = 0; i < n_iter; i++) {
    rnd.seed(i);
    int test_size = 1000;
    auto train_data = generate_text(test_size, true, rnd);
    int n_sentences = 1000;
    vector<string> inference_data;
    for (int i = 0; i < n_sentences; i++) {
      inference_data.push_back(generate_text(20, false, rnd));
    }
    set<uint32_t> unique_input_chars(train_data.begin(), train_data.end());
    int vocab_size = unique_input_chars.size() + uniform_dist_int(rnd, 4, 44);
    double character_coverage = 1 - (uniform_dist_double(rnd, 0, 1)) * 0.4;
    if (uniform_dist_int(rnd, 0, 2) == 0) {
      character_coverage = 1;
    }

    auto train_data_copy = train_data;
    BpeConfig bpe_config = {character_coverage, n_threads, {0, 1, 2, 3}};
    BPEState learned_model;
    status = learn_bpe_from_string(train_data_copy, vocab_size, "remove_it.txt", bpe_config, &learned_model);
    assert(status.ok());
    BaseEncoder applyer(learned_model, 20);

    vector<vector<string>> result_sentence_by_sentence;
    for (auto s: inference_data) {
      vector<vector<string>> encoded_subwords;
      status = applyer.encode_as_subwords({s}, &encoded_subwords);
      assert(status.ok());
      result_sentence_by_sentence.push_back(encoded_subwords[0]);
    }
    vector<vector<string>> result_parallel;
    status = applyer.encode_as_subwords(inference_data, &result_parallel);
    assert(status.ok());
    assert(result_sentence_by_sentence == result_parallel);
  }
}

void base_stress(int n_iter) {
  mt19937 rnd;
  int n_threads = 8;
  const int NUMBER_OF_SPECIAL_TOKENS_LOCAL = 4;
  Status status;
  for (int it = 0; it != n_iter; it++) {
    rnd.seed(it);
    cerr << "-------------------- new test " << it << " --------------- " << endl;
    int test_size = 1000;

    auto train_data = generate_text(test_size, true, rnd);
    set<uint32_t> unique_train_symbols(train_data.begin(), train_data.end());
    unique_train_symbols.insert(' ');
    int vocab_size = unique_train_symbols.size() + NUMBER_OF_SPECIAL_TOKENS_LOCAL + uniform_dist_int(rnd, 0, 40);

    cerr << "train_data: !" << train_data << "! (vocab_size, len): (" << vocab_size << ", " << train_data.size()
         << ")" << endl;

    double character_coverage = 1 - uniform_dist_double(rnd, 0, 1) * 0.4;
    if (uniform_dist_int(rnd, 0, 2) == 0) {
      character_coverage = 1;
    }
    auto train_data_copy = train_data;
    BpeConfig bpe_config = {character_coverage, n_threads, {0, 1, 2, 3}};
    BPEState fast_solution_model;
    status = learn_bpe_from_string(train_data_copy, vocab_size, "remove_it.txt", bpe_config, &fast_solution_model);
    assert(status.ok());
    auto slow_solution_model = learn_bpe_slow(train_data, vocab_size, "remove_it.txt", bpe_config);

    if (fast_solution_model.rules != slow_solution_model.rules
        || fast_solution_model.char2id != slow_solution_model.char2id) {
      for (auto rr: {fast_solution_model, slow_solution_model}) {
        cerr << "rules: " << endl;
        cerr << "rr.rules.size(): " << rr.rules.size() << "    rr.char2id.size(): " << rr.char2id.size() << endl;
        for (auto rule: rr.rules) {
          cerr << rule.x << " + " << rule.y << " =  " << rule.z << endl;
        }
        for (auto x: rr.char2id) {
          cerr << "id: " << x.first << "  char: " << x.second << endl;
        }
      }
    }
    assert(fast_solution_model.rules == slow_solution_model.rules);
    assert(fast_solution_model.char2id == slow_solution_model.char2id);

    BaseEncoder applyer(fast_solution_model, 1);

    auto inference_data = generate_text(test_size, false, rnd);
    cerr << "inference_data: " << inference_data << endl;
    vector<vector<int>> fast_ids_tmp;
    status = applyer.encode_as_ids({inference_data}, &fast_ids_tmp);
    auto fast_ids = fast_ids_tmp[0];
    assert(status.ok());
    vector<vector<string>> fast_pieces_tmp;
    status = applyer.encode_as_subwords({inference_data}, &fast_pieces_tmp);
    assert(status.ok());
    auto fast_pieces = fast_pieces_tmp[0];
    auto slow_results = decode_slow(inference_data, applyer);
    vector<string> slow_pieces;
    for (auto x: slow_results.pieces) {
      slow_pieces.push_back(x);
    }

    if (fast_ids != slow_results.ids) {
      cerr << "ids real: ";
      for (auto x: fast_ids) cerr << x << " ";
      cerr << endl;
      cerr << "ids slow: ";
      for (auto x: slow_results.ids) cerr << x << " ";
      cerr << endl;
      cerr << "pieces real: ";
      for (auto x: fast_pieces) cerr << x << " ";
      cerr << endl;
      cerr << "pieces slow: ";
      for (auto x: slow_results.pieces) cerr << x << " ";
      cerr << endl;
    }
    assert(fast_ids == slow_results.ids);
    assert(fast_pieces == slow_pieces);

    string fast_result_one_line;
    for (const auto &x: fast_pieces) fast_result_one_line += x;
    string slow_result_one_line = "";
    for (const auto &x: slow_pieces) slow_result_one_line += x;

    auto original_no_space = to_no_space_tokens(inference_data);
    auto fast_no_space = to_no_space_tokens(fast_result_one_line);
    auto slow_no_space = to_no_space_tokens(slow_result_one_line);

    if (fast_no_space != original_no_space) {
      cerr << "original_no_space: ";
      for (auto x: original_no_space) { cerr << x << " "; }
      cerr << endl;
      cerr << "fast_no_space: ";
      for (auto x: fast_no_space) { cerr << x << " "; }
      cerr << endl;
      cerr << "slow_no_space: ";
      for (auto x: slow_no_space) { cerr << x << " "; }
      cerr << endl;
    }
    assert(fast_no_space == original_no_space);
  }
}
}

int main(int argc, char **argv) {
  if (argc == 1) {
    vkcom::base_stress(-1);
  } else {
    int n_iter;
    if (std::string(argv[1]) == "manual") {
      vkcom::manual_test();
      return 0;
    }
    if (std::string(argv[1]) == "parallel") {
      sscanf(argv[2], "%d", &n_iter);
      vkcom::parallel_test(n_iter, 8);
      return 0;
    }
    if (std::string(argv[1]) == "base") {
      sscanf(argv[2], "%d", &n_iter);
      vkcom::base_stress(n_iter);
      return 0;
    }
    assert(false);
  }
}

