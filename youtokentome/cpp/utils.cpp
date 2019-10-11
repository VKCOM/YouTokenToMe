#include "utils.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace vkcom {
using std::string;
using std::vector;

class FileWriter : public StreamWriter {
 public:
  FileWriter(const std::string &file_name) {
    this->file_name = file_name;
    this->fout = std::ofstream(file_name, std::ios::out | std::ios::binary);
    if (fout.fail()) {
      std::cerr << "Can't open file: " << file_name << std::endl;
      assert(false);
    }
  }

  virtual int write(const char *buffer, int size) override {
    return fout.write(buffer, size);
  }

  virtual std::string name() const noexcept override {
    return file_name;
  }

 private:
  std::string file_name;
  std::ofstream fout;
};

class FileReader : public StreamReader {
 public:
  FileReader(const std::string &file_name) {
    this->file_name = file_name;
    this->fin = std::ifstream(file_name, std::ios::in | std::ios::binary);
    if (fin.fail()) {
      std::cerr << "Can't open file: " << file_name << std::endl;
      assert(false);
    }
  }

  virtual int read(const char *buffer, int size) override {
    return fin.read(buffer, size);
  }

  virtual std::string name() const noexcept override {
    return file_name;
  }

 private:
  std::string file_name;
  std::ifstream fin;
};

StreamWriter StreamWriter::open(const std::string &file_name) {
  return FileWriter(file_name);
}

StreamReader StreamReader::open(const std::string &file_name) {
  return FileReader(file_name);
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
T bin_to_int(const char *val) {
  uint32_t ret = static_cast<unsigned char>(val[0]);
  ret |= static_cast<uint32_t>(static_cast<unsigned char>(val[1])) << 8;
  ret |= static_cast<uint32_t>(static_cast<unsigned char>(val[2])) << 16;
  ret |= static_cast<uint32_t>(static_cast<unsigned char>(val[3])) << 24;
  return ret;
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
std::unique_ptr<char[]> int_to_bin(T val) {
  auto u32 = static_cast<uint32_t>(val);
  std::unique_ptr<char[]> ret(new char[4]);
  ret[0] = u32 & 0xFF;
  ret[1] = (u32 >> 8) & 0xFF;
  ret[2] = (u32 >> 16) & 0xFF;
  ret[3] = (u32 >> 24);  // no need for & 0xFF
  return std::move(ret);
}

void SpecialTokens::dump(StreamWriter &fout) {
  std::unique_ptr<char[]> unk_id_ptr(int_to_bin(unk_id)),
                          pad_id_ptr(int_to_bin(pad_id)),
                          bos_id_ptr(int_to_bin(bos_id)),
                          eos_id_ptr(int_to_bin(eos_id));
  fout.write(unk_id_ptr.get(), 4);
  fout.write(pad_id_ptr.get(), 4);
  fout.write(bos_id_ptr.get(), 4);
  fout.write(eos_id_ptr.get(), 4);
}

void SpecialTokens::load(StreamReader &fin) {
  char unk_id_bs[4], pad_id_bs[4], bos_id_bs[4], eos_id_bs[4];
  fin.read(unk_id_bs, 4);
  fin.read(pad_id_bs, 4);
  fin.read(bos_id_bs, 4);
  fin.read(eos_id_bs, 4);
  this->unk_id = bin_to_int<int>(unk_id_bs);
  this->pad_id = bin_to_int<int>(pad_id_bs);
  this->bos_id = bin_to_int<int>(bos_id_bs);
  this->eos_id = bin_to_int<int>(eos_id_bs);
}

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

size_t SpecialTokens::n_special_tokens() const {
  size_t cnt = 0;
  cnt += (unk_id != -1);
  cnt += (pad_id != -1);
  cnt += (bos_id != -1);
  cnt += (eos_id != -1);
  return cnt;
}

SpecialTokens::SpecialTokens(int pad_id, int unk_id, int bos_id, int eos_id)
    : pad_id(pad_id), unk_id(unk_id), bos_id(bos_id), eos_id(eos_id) {}

bool BPE_Rule::operator==(const BPE_Rule &other) const {
  return x == other.x && y == other.y && z == other.z;
}

BPE_Rule::BPE_Rule(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {}

void BPEState::dump(StreamWriter &fout) {
  std::unique_ptr<char[]> char2id_ptr(int_to_bin(char2id.size())),
                          rules_ptr(int_to_bin(rules.size()));
  fout.write(char2id_ptr.get(), 4);
  fout.write(rules_ptr.get(), 4);
  for (auto &s : char2id) {
    std::unique_ptr<char[]> first_ptr(int_to_bin(s.first)),
                            second_ptr(int_to_bin(s.second));
    fout.write(first_ptr.get(), 4);
    fout.write(second_ptr.get(), 4);
  }
  for (auto &rule : rules) {
    std::unique_ptr<char[]> rule_ptr(int_to_bin(rule.x));
    fout.write(rule_ptr.get(), 4);
  }
  for (auto &rule : rules) {
    std::unique_ptr<char[]> rule_ptr(int_to_bin(rule.y));
    fout.write(rule_ptr.get(), 4);
  }
  for (auto &rule : rules) {
    std::unique_ptr<char[]> rule_ptr(int_to_bin(rule.z));
    fout.write(rule_ptr.get(), 4);
  }
  special_tokens.dump(fout);
}

void BPEState::load(StreamReader &fin) {
  char2id.clear();
  rules.clear();
  char n_bs[4], m_bs[4];
  fin.read(n_bs, 4);
  fin.read(m_bs, 4);
  auto n = bin_to_int<int>(n_bs);
  auto m = bin_to_int<int>(m_bs);
  for (int i = 0; i < n; i++) {
    char inner_id_bs[4], utf32_id_bs[4];
    fin.read(inner_id_bs, 4);
    fin.read(utf32_id_bs, 4);
    auto inner_id = bin_to_int<uint32_t>(inner_id_bs);
    auto utf32_id = bin_to_int<uint32_t>(utf32_id_bs);
    char2id[inner_id] = utf32_id;
  }
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> rules_xyz(m);
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < m; i++) {
      char val[4];
      fin.read(val, 4);
      uint32_t *element;
      switch (j) {
        case 0:
          element = &std::get<0>(rules_xyz[i]);
        case 1:
          element = &std::get<1>(rules_xyz[i]);
        case 2:
          element = &std::get<2>(rules_xyz[i]);
      }
      *element = bin_to_int<uint32_t>(val);
    }
  }
  for (int i = 0; i < m; i++) {
    rules.emplace_back(std::get<0>(rules_xyz[i]), std::get<1>(rules_xyz[i]), std::get<2>(rules_xyz[i]));
  }
  special_tokens.load(fin);
}

BpeConfig::BpeConfig(double _character_coverage, int _n_threads,
                     const SpecialTokens &_special_tokens)
    : character_coverage(_character_coverage),
      n_threads(_n_threads),
      special_tokens(_special_tokens) {}

vector<string> read_lines_from_stdin(size_t batch_limit, size_t *processed) {
  vector<string> sentences;
  string s;
  while (*processed < batch_limit && getline(std::cin, s)) {
    *processed += s.size();
    sentences.push_back(std::move(s));
  }
  return sentences;
}

}  // namespace vkcom
