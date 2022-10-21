#pragma once

#include "../../youtokentome/cpp/third_party/flat_hash_map.h"
#include "../../youtokentome/cpp/bpe.h"

namespace vkcom {

flat_hash_map<uint32_t, uint32_t>
compute_alphabet_helper(const flat_hash_map<uint32_t, uint64_t> &char_cnt,
                        uint64_t data_len,
                        flat_hash_set<uint32_t> &removed_chars,
                        const BpeConfig &bpe_config);

Status learn_bpe_from_string(std::string &text_utf8,
                             int n_tokens,
                             const std::string &output_file,
                             BpeConfig bpe_config,
                             BPEState *bpe_state);

} // namespace vkcom
