from enum import Enum
from functools import wraps
from typing import BinaryIO, List, Optional, Union

import _youtokentome_cython


class OutputType(Enum):
    ID = 1
    SUBWORD = 2


class BPE:
    def __init__(self, model: Union[str, BinaryIO], n_threads: int = -1):
        own_obj = isinstance(model, str)
        if own_obj:
            model = open(model, "rb")
        try:
            self.bpe_cython = _youtokentome_cython.BPE(
                model_fobj=model, n_threads=n_threads
            )
        finally:
            if own_obj:
                model.close()

    @staticmethod
    def train(
        data: str,
        model: Optional[Union[str, BinaryIO]],
        vocab_size: int,
        coverage: float = 1.0,
        n_threads: int = -1,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> "BPE":
        own_obj = isinstance(model, str)
        if own_obj:
            model = open(model, "wb")
        try:
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
        finally:
            if own_obj:
                model.close()

        return BPE(model=model, n_threads=n_threads)

    def encode(
        self,
        sentences: List[str],
        output_type: OutputType = OutputType.ID,
        bos: bool = False,
        eos: bool = False,
        reverse: bool = False,
    ) -> Union[List[List[int]], List[List[str]]]:
        if not isinstance(output_type, OutputType):
            raise TypeError(
                "parameter output_type must be youtokentome.OutputType, not %s}" % str(type(output_type))
            )

        output_type_str = "id" if output_type == OutputType.ID else "subword"
        return self.bpe_cython.encode(
            sentences=sentences,
            output_type=output_type_str,
            bos=bos,
            eos=eos,
            reverse=reverse,
        )

    def save(self, where: Union[str, BinaryIO]):
        """
        Write the model to FS or any writeable file object.

        :param where: FS path or writeable file object.
        :return: None
        """
        own_obj = isinstance(where, str)
        if own_obj:
            where = open(where, "wb")
        try:
            self.bpe_cython.save(where=where)
        finally:
            if own_obj:
                where.close()

    def vocab_size(self) -> int:
        return self.bpe_cython.vocab_size()

    def vocab(self) -> List[str]:
        return self.bpe_cython.vocab()

    def subword_to_id(self, subword: str) -> int:
        return self.bpe_cython.subword_to_id(subword)

    def id_to_subword(self, id: int) -> str:
        return self.bpe_cython.id_to_subword(id)

    def decode(self, ids: List[int]) -> str:
        return self.bpe_cython.decode(ids)
