from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from libcpp cimport bool
import os
from pathlib import Path
from typing import Collection


cdef extern from "bpe.h" namespace "vkcom":

    cdef cppclass SpecialTokens:
        int pad_id
        int unk_id
        int bos_id
        int eos_id

    cdef cppclass BpeConfig:
        double character_coverage
        int n_threads
        SpecialTokens special_tokens

    cdef cppclass Status:
        int code
        string message


cdef extern from "bpe.h" namespace "vkcom":
    Status train_bpe(const string &source_path, const string& model_path, int vocab_size, const BpeConfig& bpe_config)

cdef extern from "bpe.h" namespace "vkcom":
    cdef cppclass BaseEncoder:
        BaseEncoder(const string& model_path, int n_threads, Status* status)

        Status encode_as_ids(const vector[string] &sentences, vector[vector[int]]* ids, bool bos, bool eos, bool reverse, double dropout_prob) const
        Status encode_as_subwords(const vector[string]& sentences, vector[vector[string]]* subwords, bool bos, bool eos, bool reverse, double dropout_prob) const

        Status encode_cli(string output_type, bool stream, bool bos, bool eos, bool reverse, double dropout_prob) const

        Status decode_cli(const unordered_set[int]* ignore_ids) const

        void vocab_cli(bool verbose) const

        Status id_to_subword(int id, string* subword) const

        int subword_to_id(const string &subword) const
        Status decode(const vector[vector[int]]& ids, vector[string]* output, const unordered_set[int]* ignore_ids) const
        int vocab_size() const
        vector[string] vocabulary() const


cdef class BPE:
    cdef BaseEncoder* encoder

    def __dealloc__(self):
        del self.encoder

    def __init__(self, model_path, n_threads=-1):
        cdef Status status
        self.encoder = new BaseEncoder(model_path.encode(), n_threads, &status)
        if status.code != 0:
            raise ValueError(status.message.decode())

    @staticmethod
    def train(data,
              model,
              vocab_size,
              coverage=1.0,
              n_threads=-1,
              pad_id=0,
              unk_id=1,
              bos_id=2,
              eos_id=3):

        cdef BpeConfig bpe_config
        bpe_config.character_coverage = coverage
        bpe_config.n_threads = n_threads
        bpe_config.special_tokens.pad_id = pad_id
        bpe_config.special_tokens.unk_id = unk_id
        bpe_config.special_tokens.bos_id = bos_id
        bpe_config.special_tokens.eos_id = eos_id

        cdef Status status = train_bpe(data.encode(), model.encode(), vocab_size, bpe_config)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def encode(self, sentences, output_type, bos, eos, reverse, dropout_prob):
        cdef vector[string] s
        cdef vector[vector[string]] ret_subwords
        cdef vector[vector[int]] ret_ids
        cdef Status status
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError("dropout_prob value must be in the range [0, 1]. Current value of dropout_prob = " + str(dropout_prob))
        if output_type == 'id':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse, dropout_prob)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                return ret_ids[0]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse, dropout_prob)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return ret_ids
        elif output_type == 'subword':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse, dropout_prob)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                assert len(ret_subwords) == 1
                return [piece.decode() for piece in ret_subwords[0]]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse, dropout_prob)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return [[piece.decode() for piece in sentence] for sentence in ret_subwords]
        else:
            raise ValueError('output_type must be equal to "id" or "subword"')

    def subword_to_id(self, subword):
        return self.encoder.subword_to_id(subword.encode())

    def id_to_subword(self, id):
        cdef string subword
        cdef Status status = self.encoder.id_to_subword(id, &subword)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return subword.decode()

    def decode(self, ids, ignore_ids):

        if not isinstance(ids, list):
            raise TypeError(
                "{} is not a list instance".format(type(ids))
            )

        if not isinstance(ignore_ids, Collection) and ignore_ids is not None:
            raise TypeError(
                "{} is not a Collection instance".format(type(ignore_ids))
            )

        if len(ids) > 0 and isinstance(ids[0], int):
            ids = [ids]
        if ignore_ids is None:
            ignore_ids = set()

        cdef vector[string] sentences
        cdef unordered_set[int] c_ignore_ids = unordered_set[int](ignore_ids)
        cdef Status status = self.encoder.decode(ids, &sentences, &c_ignore_ids)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return [sentence.decode() for sentence in sentences]

    def vocab_size(self):
        return self.encoder.vocab_size();

    def vocab(self):
        cdef vector[string] vocab = self.encoder.vocabulary()
        return [token.decode() for token in vocab]

    def encode_cli(self, output_type, stream, bos, eos, reverse, dropout_prob):
        cdef Status status = self.encoder.encode_cli(output_type.encode(), stream, bos, eos, reverse, dropout_prob)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def decode_cli(self, ignore_ids):
        if ignore_ids is None:
            ignore_ids = set()
        cdef unordered_set[int] c_ignore_ids = unordered_set[int](ignore_ids)
        cdef Status status = self.encoder.decode_cli(&c_ignore_ids)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def vocab_cli(self, verbose):
        self.encoder.vocab_cli(verbose)

