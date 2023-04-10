#include "utils.h"

#include <fstream>
#include <string>
#include <vector>

namespace vkcom {

Status::Status(int code, std::string message) : code(code), message(std::move(message)) {}

const std::string &Status::error_message() const { return message; }

bool Status::ok() const { return code == 0; }

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

std::vector<std::string> read_all_lines(std::istream& stream) {
  std::vector<std::string> sentences;
  std::string s;
  while (std::getline(stream, s)) {
    sentences.push_back(std::move(s));
  }
  return sentences;
}

std::vector<std::string> read_lines(std::istream& stream, uint64_t batch_limit, uint64_t *processed) {
  std::vector<std::string> sentences;
  std::string s;
  while (*processed < batch_limit && std::getline(stream, s)) {
    *processed += s.size();
    sentences.push_back(std::move(s));
  }
  return sentences;
}

Status fast_read_file_utf8(const std::string &file_name, std::string *file_content) {
  static const int buf_size = 1000000;
  *file_content = "";
  // TODO: use ifstream and seekg+tellg+seekg to reserve
  auto fin = fopen(file_name.data(), "rb");
  if (fin == nullptr) {
    return Status(1, "Failed to open file: " + file_name);
  }
  while (true) {
    uint64_t cur_size = file_content->size();
    file_content->resize(cur_size + buf_size);
    int buf_len = fread((void *)(file_content->data() + cur_size), 1, buf_size, fin);
    if (buf_len < buf_size) {
      file_content->resize(file_content->size() - (buf_size - buf_len));
      fclose(fin);
      return Status();
    }
  }
}

} // namespace vkcom
