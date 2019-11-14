from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from libcpp cimport bool
import os
from pathlib import Path


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

        Status encode_as_ids(const vector[string] &sentences, vector[vector[int]]* ids, bool bos, bool eos, bool reverse) const
        Status encode_as_subwords(const vector[string]& sentences, vector[vector[string]]* subwords, bool bos, bool eos, bool reverse) const

        Status encode_cli(string output_type, bool stream, bool bos, bool eos, bool reverse) const

        Status decode_cli() const

        void vocab_cli(bool verbose) const

        Status id_to_subword(int id, string* subword) const

        int subword_to_id(const string &subword) const
        Status decode(const vector[vector[int]]& ids, vector[string]* output, unordered_set[int]& ignore_ids) const
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

    def encode(self, sentences, output_type, bos, eos, reverse):
        cdef vector[string] s
        cdef vector[vector[string]] ret_subwords
        cdef vector[vector[int]] ret_ids
        cdef Status status
        if output_type == 'id':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                assert len(s) == 1
                status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                return ret_ids[0]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_ids(s, &ret_ids, bos, eos, reverse)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return ret_ids
        elif output_type == 'subword':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                assert len(ret_subwords) == 1
                return [piece.decode() for piece in ret_subwords[0]]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_subwords(s, &ret_subwords, bos, eos, reverse)
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
        assert isinstance(
            ids, list
        ), "ids have to be a list instance but {} found".format(type(list))
        assert (
            isinstance(ignore_ids, set)
            or isinstance(ignore_ids, list)
            or ignore_ids is None
        ), "ids have to be a list instance or set instance, but {} found".format(
            type(ignore_ids)
        )
        if len(ids) > 0 and isinstance(ids[0], int):
            ids = [ids]
        if ignore_ids is None:
            ignore_ids = set([])
        elif isinstance(ignore_ids, list):
            ignore_ids = set(ignore_ids)
        cdef vector[string] sentences
        cdef Status status = self.encoder.decode(ids, &sentences, ignore_ids)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return [sentence.decode() for sentence in sentences]

    def vocab_size(self):
        return self.encoder.vocab_size();

    def vocab(self):
        cdef vector[string] vocab = self.encoder.vocabulary()
        return [token.decode() for token in vocab]

    def encode_cli(self, output_type, stream, bos, eos, reverse):
        cdef Status status = self.encoder.encode_cli(output_type.encode(), stream, bos, eos, reverse)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def decode_cli(self):
        cdef Status status = self.encoder.decode_cli()
        if status.code != 0:
            raise ValueError(status.message.decode())

    def vocab_cli(self, verbose):
        self.encoder.vocab_cli(verbose)

