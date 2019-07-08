#pragma once

#include "utils.h"

namespace vkcom {
std::string encode_utf8(const std::vector<uint32_t> &utext);

std::vector<uint32_t> decode_utf8(const char *begin, const char *end);

std::vector<uint32_t> decode_utf8(const std::string& utf8_text);

} // namespace vkcom



