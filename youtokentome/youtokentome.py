from enum import Enum
from typing import List, Union, Optional, Collection

import _youtokentome_cython


class OutputType(Enum):
    ID = 1
    SUBWORD = 2


class BPE:
    def __init__(self, model: str, n_threads: int = -1):
        self.bpe_cython = _youtokentome_cython.BPE(
            model_path=model, n_threads=n_threads
        )

    @staticmethod
    def train(
        data: str,
        model: str,
        vocab_size: int,
        coverage: float = 1.0,
        n_threads: int = -1,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> "BPE":
        _youtokentome_cython.BPE.train(
            data=data,
            model=model,
            vocab_size=vocab_size,
            n_threads=n_threads,
            coverage=coverage,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )

        return BPE(model=model, n_threads=n_threads)

    def encode(
        self,
        sentences: List[str],
        output_type: OutputType = OutputType.ID,
        bos: bool = False,
        eos: bool = False,
        reverse: bool = False,
        dropout_prob: float = 0,
    ) -> Union[List[List[int]], List[List[str]]]:
        if not isinstance(output_type, OutputType):
            raise TypeError(
                "parameter output_type must be youtokentome.OutputType, not %s}"
                % str(type(output_type))
            )

        output_type_str = "id" if output_type == OutputType.ID else "subword"
        return self.bpe_cython.encode(
            sentences=sentences,
            output_type=output_type_str,
            bos=bos,
            eos=eos,
            reverse=reverse,
            dropout_prob=dropout_prob,
        )

    def vocab_size(self) -> int:
        return self.bpe_cython.vocab_size()

    def vocab(self) -> List[str]:
        return self.bpe_cython.vocab()

    def subword_to_id(self, subword: str) -> int:
        return self.bpe_cython.subword_to_id(subword)

    def id_to_subword(self, id: int) -> str:
        return self.bpe_cython.id_to_subword(id)

    def decode(
        self, ids: List[int], ignore_ids: Optional[Collection[int]] = None
    ) -> str:
        return self.bpe_cython.decode(ids, ignore_ids)
