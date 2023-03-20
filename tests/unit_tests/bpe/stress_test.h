#pragma once

#include "../../youtokentome/cpp/third_party/flat_hash_map/flat_hash_map.h"
#include "../../youtokentome/cpp/utils.h"

namespace vkcom {
flat_hash_map<uint32_t, uint32_t>
compute_alphabet(const std::vector<uint32_t> &data,
                 flat_hash_set<uint32_t> &removed_chars,
                 const BpeConfig &bpe_config);

void remove_rare_chars(std::vector<uint32_t> &data, const flat_hash_set<uint32_t> &removed_chars);

Status learn_bpe_from_string(std::string &text_utf8,
                             int n_tokens,
                             const std::string &output_file,
                             BpeConfig bpe_config,
                             BPEState *bpe_state);

void utf8_to_chars(uint32_t x, std::back_insert_iterator<std::string> it);

uint32_t chars_to_utf8(const char *begin, size_t size, size_t *utf8_len);

}
