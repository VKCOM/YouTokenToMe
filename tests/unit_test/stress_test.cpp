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

#include <chrono>
#include <thread>

namespace vkcom {
using namespace std;

extern int alive_tokens;

using char32=uint32_t;

BPEState learn_bpe_word_level_slow(const string &text_utf8, int n_token, string, BpeConfig bpe_config) {
  cerr << "#### SLOW ##############" << endl;
  auto row_data = decode_utf8(text_utf8.data(), text_utf8.data() + text_utf8.size());

  vector<vector<uint32_t>> splited;

  for (auto &ch: row_data) {
    if (is_space(ch)) {
      ch = SPACE_TOKEN;
    }
  }

  for (; !row_data.empty() && is_space(row_data.back()); row_data.pop_back());

  ska::flat_hash_set<uint32_t> removed_chars;
  auto char2id = compute_alphabet(row_data, removed_chars, bpe_config);
  remove_rare_chars(row_data, removed_chars);

  ska::flat_hash_map<uint32_t, uint32_t> id2char;
  for (auto x: char2id) {
    id2char[x.second] = x.first;
  }

  int used_ids = bpe_config.special_tokens.n_special_tokens() + char2id.size();

  for (int i = 0; i < (int) row_data.size();) {
    for (; i < (int) row_data.size() && is_space(row_data[i]); i++);
    if (i == (int) row_data.size()) {
      break;
    }
    splited.emplace_back();
    splited.back().push_back(SPACE_TOKEN);
    for (; i < (int) row_data.size() && !is_space(row_data[i]); i++) {
      if (char2id.count(row_data[i])) {
        splited.back().push_back(row_data[i]);
      }
    }
  }
  vector<vector<int>> coded;

  for (const auto &v: splited) {
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
    vector<int> res;
    for (auto token_id: recipe[x]) res.push_back(token_id);
    for (auto token_id: recipe[y]) res.push_back(token_id);
    return res;
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
      auto rz = get_recipe(x, y);

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

DecodeResult decode_word_level_slow(const string &text_utf8, const BaseEncoder &bpe_applyer) {

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
        auto tmp = recipe.at(u.val);
        vector<uint32_t> utmp;
        for (auto ch: tmp) {
          assert(id2char.count(ch));
          utmp.push_back(id2char.at(ch));
        }
        pieces.push_back(encode_utf8(utmp));
      }
    }
  }

  return {ids, pieces};
}

string generate_text(int n_limit, bool flag_train) {
  string sigma = flag_train ? "abc " : "abcd ";
  vector<uint32_t> a;
  int n = rand() % 20 + 1;
  n = rand() % 1000 + 1;
  n = min(n, n_limit);
  string row_data;
  row_data.push_back(sigma[0]);

  auto m_pb = [&](char ch) {
    row_data.push_back(ch);
  };

  for (; (int) row_data.size() < n;) {
    if (rand() % 2) {
      m_pb(sigma[rand() % sigma.size()]);
    } else {
      int l = rand() % 5 + 2;
      int seg = rand() % 4 + 1;
      vector<uint32_t> tmp;
      for (int i = 0; i < seg; i++) {
        m_pb(sigma[rand() % sigma.size()]);
      }
      for (int i = 0; i < l; i++) {
        for (auto ch: tmp) {
          m_pb(ch);
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
  string trn_data = "baba baaab";
  string inf_data = "d d";
  int n_tokens = 2 + 2 + 5;

  auto trn_data_copy = trn_data;
  SpecialTokens ff = {0, 1, 2, 3};
  BpeConfig bpe_config = {1.0, 1, ff};

  auto r1 = learn_bpe_from_string(trn_data_copy, n_tokens, "remove_it.txt", bpe_config);
  auto r2 = learn_bpe_word_level_slow(trn_data, n_tokens, "remove_it.txt", bpe_config);
  assert(r1.rules == r2.rules);
  assert(r1.char2id == r2.char2id);

  BaseEncoder applyer(r1, 1);
  auto ids = applyer.encode_as_ids({inf_data})[0];
  auto d2 = decode_word_level_slow(inf_data, applyer);
  assert(ids == d2.ids);
}

vector<uint32_t> to_no_space_tokens(string a) {
  auto a_tokens = decode_utf8(a.data(), a.data() + a.size());
  int cur = 0;
  for (auto ch: a_tokens) {
    if (!is_space(ch)) {
      a_tokens[cur++] = ch;
    }
  }
  a_tokens.resize(cur);
  return a_tokens;
}

void parallel_test(int n_threads) {
  int test_size = 1000;
  auto train_data = generate_text(test_size, true);
  int n_sentences = 1000;
  vector<string> inference_data;
  for (int i = 0; i < n_sentences; i++) {
    inference_data.push_back(generate_text(20, false));
  }
  set<uint32_t> tmp(train_data.begin(), train_data.end());
  int vocab_size = tmp.size() + 4 + rand() % 40;
  double character_coverage = 1 - (rand() * 1.0 / RAND_MAX) * 0.4;
  if (rand() % 2 == 0) {
    character_coverage = 1;
  }

  auto train_data_copy = train_data;
  BpeConfig bpe_config = {character_coverage, n_threads, {0, 1, 2, 3}};
  auto r1 = learn_bpe_from_string(train_data_copy, vocab_size, "remove_it.txt", bpe_config);
  BaseEncoder applyer(r1, 20);

  vector<vector<string>> res1;
  for (auto s: inference_data) {
    res1.push_back(applyer.encode_as_subwords({s})[0]);
  }
  auto res2 = applyer.encode_as_subwords(inference_data);
  assert(res1 == res2);

}

mt19937 rnd;

void stress_full(int n_iter) {
  bool stress_all = true;
  int n_threads = 8;
  if (stress_all) {
    manual_test();
  }

  const int NUMBER_OF_SPECIAL_TOKENS_LOCAL = 4;
  int gg = 0;
  for (int it = 0; it != n_iter; it++) {
    auto st_time = chrono::steady_clock::now();
    gg++;
    srand(it);
    cerr << "-------------------- new test " << it << ", " << gg << " --------------- " << endl;
    int test_size = 1000;

    if (it % 50 == 49 && stress_all) {
      parallel_test(n_threads);
    }

    auto train_data = generate_text(test_size, true);
    set<uint32_t> tmp(train_data.begin(), train_data.end());
    tmp.insert(' ');
    int vocab_size = tmp.size() + NUMBER_OF_SPECIAL_TOKENS_LOCAL + rand() % 40;

    cerr << "train_data: !" << train_data << "! (vocab_size, len): (" << vocab_size << ", " << train_data.size()
         << ")" << endl;

    double character_coverage = 1 - (rand() * 1.0 / RAND_MAX) * 0.4;
    if (rand() % 2 == 0) {
      character_coverage = 1;
    }
    auto train_data_copy = train_data;
    BpeConfig bpe_config = {character_coverage, n_threads, {0, 1, 2, 3}};
    auto r1 = learn_bpe_from_string(train_data_copy, vocab_size, "remove_it.txt", bpe_config);
    auto r2 = learn_bpe_word_level_slow(train_data, vocab_size, "remove_it.txt", bpe_config);

    if (r1.rules != r2.rules || r1.char2id != r2.char2id) {
      cerr << " fast slow " << endl;
      for (auto rr: {r1, r2}) {
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

    assert(r1.rules == r2.rules);
    assert(r1.char2id == r2.char2id);

    BaseEncoder applyer(r1, 1);

    auto inference_data = generate_text(test_size, false);
    cerr << "inference_data: " << inference_data << endl;
    auto d1_ids = applyer.encode_as_ids({inference_data})[0];
    auto d1_pieces = applyer.encode_as_subwords({inference_data})[0];
    auto d2 = decode_word_level_slow(inference_data, applyer);
    vector<string> d2_pieces;
    for (auto x: d2.pieces) {
      d2_pieces.push_back(x);
    }

    if (d1_ids != d2.ids) {
      cerr << "ids real: ";
      for (auto x: d1_ids) cerr << x << " ";
      cerr << endl;
      cerr << "ids slow: ";
      for (auto x: d2.ids) cerr << x << " ";
      cerr << endl;
      cerr << "pieces real: ";
      for (auto x: d1_pieces) cerr << x << " ";
      cerr << endl;
      cerr << "pieces slow: ";
      for (auto x: d2.pieces) cerr << x << " ";
      cerr << endl;
    }
    assert(d1_ids == d2.ids);
    assert(d1_pieces == d2_pieces);

    string d1_one_line;
    for (const auto &x: d1_pieces) d1_one_line += x;
    string d2_one_line = "";
    for (const auto &x: d2_pieces) d2_one_line += x;

    auto c0 = to_no_space_tokens(inference_data);
    auto c1 = to_no_space_tokens(d1_one_line);
    auto c2 = to_no_space_tokens(d2_one_line);

    if (c1 != c0) {
      cerr << "c0: ";
      for (auto x: c0) { cerr << x << " "; }
      cerr << endl;
      cerr << "c1: ";
      for (auto x: c1) { cerr << x << " "; }
      cerr << endl;
      cerr << "c2: ";
      for (auto x: c2) { cerr << x << " "; }
      cerr << endl;
    }

    assert(c1 == c0);
    cerr << "passed!" << endl;
    cerr << "one test time: "
         << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - st_time).count() << endl;
  }
}
}

int main(int argc, char **argv) {
  if (argc == 1) {
    vkcom::stress_full(-1);
  } else {
    int n_iter;;
    sscanf(argv[1], "%d", &n_iter);
    vkcom::stress_full(n_iter);
  }
}

