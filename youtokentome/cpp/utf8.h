#pragma once

#include "utils.h"

namespace vkcom {

constexpr static uint32_t INVALID_UNICODE = 0x0fffffff;

uint32_t chars_to_utf8(const char* begin, uint64_t size, uint64_t* utf8_len);

void utf8_to_chars(const uint32_t x, const std::back_insert_iterator<std::string> it);

std::string encode_utf8(const std::vector<uint32_t> &utext);

std::vector<uint32_t> decode_utf8(const char *begin, const char *end);

std::vector<uint32_t> decode_utf8(const std::string &utf8_text);




} // namespace vkcom



