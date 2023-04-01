#pragma once

#include <string>
#include <vector>
#include <unordered_set>

namespace vkcom::wordpiece {

Status encode_as_ids(const std::string &text_path,
                     const std::string& vocab_path, std::vector<int> *ids);

Status encode_as_ids(const std::string &text,
                     const std::vector<std::string>& vocab, std::vector<int> *ids);

Status encode_as_subwords(const std::string &text_path,
                          const std::string& vocab_path,
                          std::vector<std::string> *subwords);

Status encode_as_subwords(const std::string &text,
                          const std::vector<std::string>& vocab,
                          std::vector<std::string> *subwords);

Status decode(const std::vector<int>& ids,
              const std::vector<std::string>& vocab,
              std::vector<std::string> *subwords,
              const std::unordered_set<int> *ignore_ids);

} // namespace vkcom::wordpiece