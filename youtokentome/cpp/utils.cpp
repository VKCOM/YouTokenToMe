#include "utils.h"
#include <fstream>
#include <string>
#include <vector>

namespace vkcom {

void SpecialTokens::dump(std::ofstream &fout) {
  fout << unk_id << " " << pad_id << " " << bos_id << " " << eos_id << std::endl;
}

void SpecialTokens::load(std::ifstream &fin) { fin >> unk_id >> pad_id >> bos_id >> eos_id; }

uint32_t SpecialTokens::max_id() const {
  int ret = 0;
  ret = std::max(ret, unk_id);
  ret = std::max(ret, pad_id);
  ret = std::max(ret, bos_id);
  ret = std::max(ret, eos_id);
  return ret;
}

bool SpecialTokens::taken_id(int id) const {
  return id == unk_id || id == pad_id || id == bos_id || id == eos_id;
}

uint64_t SpecialTokens::n_special_tokens() const {
  uint64_t cnt = 0;
  cnt += (unk_id != -1);
  cnt += (pad_id != -1);
  cnt += (bos_id != -1);
  cnt += (eos_id != -1);
  return cnt;
}

SpecialTokens::SpecialTokens(int pad_id, int unk_id, int bos_id, int eos_id)
 : pad_id(pad_id), unk_id(unk_id), bos_id(bos_id), eos_id(eos_id) {}

bool MergeRule::operator==(const MergeRule &other) const {
  return x == other.x && y == other.y && z == other.z;
}

bool is_space(uint32_t ch) { return (ch < 256 && isspace(ch)) || (ch == SPACE_TOKEN); }

std::vector<std::string> read_lines_from_stdin(uint64_t batch_limit, uint64_t *processed) {
  std::vector<std::string> sentences;
  std::string s;
  while (*processed < batch_limit && getline(std::cin, s)) {
    *processed += s.size();
    sentences.push_back(std::move(s));
  }
  return sentences;
}

Status::Status(int code, std::string message) : code(code), message(std::move(message)) {}

const std::string &Status::error_message() const { return message; }
bool Status::ok() const { return code == 0; }

} // namespace vkcom
