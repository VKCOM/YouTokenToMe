![PyPI](https://img.shields.io/pypi/v/youtokentome.svg)
[![Downloads](https://pepy.tech/badge/youtokentome)](https://pepy.tech/project/youtokentome)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![GitHub](https://img.shields.io/github/license/vkcom/youtokentome.svg)
[![Build Status](https://travis-ci.org/VKCOM/YouTokenToMe.svg?branch=master)](https://travis-ci.org/VKCOM/YouTokenToMe)

# YouTokenToMe 

YouTokenToMe is an unsupervised text tokenizer focused on computational efficiency. It currently implements fast Byte Pair Encoding (BPE) [[Sennrich et al.](https://www.aclweb.org/anthology/P16-1162)].
Our implementation is much faster in training and tokenization than [Hugging Face](https://github.com/huggingface/tokenizers), [fastBPE](https://github.com/glample/fastBPE)
 and [SentencePiece](https://github.com/google/sentencepiece). In some test cases, it is 90 times faster.
  Check out our [benchmark](benchmark.md) results.
  
Key advantages:

* Multithreading for training and tokenization
* The algorithm has  `O(N)` complexity, where `N` is the length of training data
* Highly efficient implementation in C++
* Python wrapper and command-line interface

Extra features:
* BPE-dropout (as described in [Provilkov et al, 2019](https://arxiv.org/abs/1910.13267))

As well as in the algorithm from the original paper, ours does not consider tokens 
that cross word boundaries. Just like in [SentencePiece](https://github.com/google/sentencepiece), all space symbols were replaced by meta symbol "▁" (U+2581). It allows sequences of tokens to be converted back to text and for word boundaries to be restored.

For example, the phrase ```Blazingly fast tokenization!``` can be tokenized into

`['▁Bl', 'az', 'ingly', '▁fast', '▁token', 'ization', '!']`

## Installation

```bash
pip install youtokentome
```
## Python interface 

### Example
Let's start with a self-contained example. 

```python
import random

import youtokentome as yttm

train_data_path = "train_data.txt"
model_path = "example.model"

# Generating random file with training data
# 10000 lines with 100 characters in each line
n_lines = 10000
n_characters = 100
with open(train_data_path, "w") as fout:
    for _ in range(n_lines):
        print("".join([random.choice("abcd ") for _ in range(n_characters)]), file=fout)

# Generating random text
test_text = "".join([random.choice("abcde ") for _ in range(100)])

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)

# Two types of tokenization
print(bpe.encode([test_text], output_type=yttm.OutputType.ID))
print(bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD))
```

&nbsp;
### Training model
```python
youtokentome.BPE.train(data, model, vocab_size, coverage, n_threads=-1, pad_id=0, unk_id=1, bos_id=2, eos_id=3)
```
Trains BPE model and saves to file.

**Args:**
 
* `data`: string, path to file with training data
* `model`: string, path to where the trained model will be saved
* `vocab_size`: int, number of tokens in the final vocabulary
* `coverage`: float, fraction of characters covered by the model. Must be in the range [0, 1]. A good value to use is about 0.9999.
* `n_threads`: int, number of parallel threads used to run. If -1 is passed, then all available threads are going to be used. Note that the number of threads is limited by 8 (see [benchmark](benchmark.md#number-of-threads)).
* `pad_id`: int, reserved id for padding
* `unk_id`: int, reserved id for unknown symbols
* `bos_id`: int, reserved id for begin of sentence token
* `eos_id`: int, reserved id for end of sentence token
 
**Returns**: Class `youtokentome.BPE` with the loaded model.
 

&nbsp;

### Model loading

```python
youtokentome.BPE(model, n_threads=-1)
```

Class constructor. Loads the trained model.

* `model`: string, path to the trained model
* `n_threads`: int, number of parallel threads used to run. 
    If equal to -1, then the maximum number of threads available will be used.
 
&nbsp;
  
### Methods
Class `youtokentome.BPE` has the following methods:
#### encode 
```python
encode(self, sentences, output_type=yttm.OutputType.ID, bos=False, eos=False, reverse=False, dropout_prob=0)
```

**Args:**
  
* `sentences`: list of strings, sentences for tokenization.
* `output_type`: enum, sentence can be tokenized to ids or subwords. Use `OutputType.ID` for ids and `OutputType.SUBWORD` for subwords.
* `bos`: bool, if True then token “beginning of sentence” will be added
* `eos`: bool, if True then token “end of sentence” will be added
* `reverse`: bool, if True the output sequence of tokens will be reversed
* `dropout_prob`: float, BPE-dropout probability (the probability of a merge being dropped). Must be in the range [0, 1].

  
**Returns:** If `output_type` is equal to `youtokentome.OutputType.ID` or `youtokentome.OutputType.SUBWORD` 
 then a list of lists of integers or list of lists of strings will be returned
respectively.

&nbsp;
#### vocab

```python
vocab(self)
```

**Returns:** A list `vocab_size` strings. The i-th string in the list corresponds
 to i-th subword.
 
&nbsp;
#### vocab_size

```python
vocab_size(self)
```

**Returns:** int. Size of vocabulary.

&nbsp;
#### subword_to_id

```python
subword_to_id(self, subword)
```
**Args:**
* `subword`: string. 

**Returns:** 
Integer from the range [0, vocab_size-1]. Id of subword or,
 if there is no such subword in the vocabulary, `unk_id` will be 
returned.

&nbsp;
#### id_to_subword 

```python
id_to_subword(self, id)
```
**Args:**
* `id`: int, must be in the range [0, vocab_size-1]

**Returns:** string. Subword from vocabulary by id.
  
&nbsp;
#### decode 
```python
decode(self, ids, ignore_ids=None)
```  
Convert each id to subword and concatenate with space symbol.

**Args:**

  * `ids`: list of lists of integers. All integers must be in the range [0, vocab_size-1]
  * `ignore_ids`: collection of integers. These indices would be ignored during the decoding. All integers must be in the range [0, vocab_size-1] [default: None]

  
**Returns:** List of strings.  
 
## Command line interface

### Example 

```bash
$ yttm bpe --data TRAINING_DATA_FILE --model OUTPUT_MODEL_FILE --vocab_size 2000
$ yttm encode --model OUTPUT_MODEL_FILE --output_type subword < TEST_DATA_FILE > ENCODED_DATA 
```


### Supported commands

`YouTokenToMe` supports the following commands:

```
$ yttm --help

Usage: yttm [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  bpe     Train BPE model.
  decode  Decode ids to text.
  encode  Encode text to ids or subwords.
  vocab   Print list of learned subwords.
```

Command `bpe` allows you to train Byte Pair Encoding model based on a text file.

```
$ yttm bpe --help

Usage: yttm bpe [OPTIONS]

  Train BPE model.

Options:
  --data PATH           Training data file path.  [required]
  --model PATH          Output model file path.  [required]
  --vocab_size INTEGER  Number of tokens in the final vocabulary.  [required]
  --coverage FLOAT      Fraction of characters covered by the model.  [default: 1.0]
  --n_threads INTEGER   Number of threads.  [default: -1]
  --pad_id INTEGER      Padding token id.  [default: 0]
  --unk_id INTEGER      Unknown token id.  [default: 1]
  --bos_id INTEGER      'Begin of sentence' token id.  [default: 2]
  --eos_id INTEGER      'End of sentence' token id.  [default: 3]
  --help                Show this message and exit.
```


Apply BPE encoding for a corpus of sentences. Use `stdin` for input and `stdout` for output.

By default, encoding works in parallel using `n_threads` threads. Number of threads is limited by
8 (see [benchmark](benchmark.md#number-of-threads)).

With the `--stream` option, `--n_threads` will be ignored and all sentences will be processed one by one.
 Each sentence will be tokenized and written to the `stdout` before the next sentence is read.


```
$ yttm encode --help

Usage: yttm encode [OPTIONS]

  Encode text to ids or subwords.

Options:
  --model PATH         Path to file with learned model.  [required]
  --output_type TEXT   'id' or 'subword'.  [required]
  --n_threads INTEGER  Number of threads.  [default: -1]
  --bos                Add tab 'begin of sentence'.
  --eos                Add tab 'end of sentence'.
  --reverse            Reverse output sequence of tokens.
  --stream             Process each line before reading the next one.
  --dropout_prob       BPE-dropout probability (the probability of a merge being dropped). [default: 0]
  --help               Show this message and exit.
```

Print vocabulary. This can be useful for understanding the model.

```
$ yttm vocab --help

Usage: yttm vocab [OPTIONS]

  Print list of learned subwords.

Options:
  --model PATH  Path to file with learned model.  [required]
  --verbose     Add merging rules.
  --help        Show this message and exit.
```

Convert ids back to text. Use `stdin` for input and `stdout` for output.

```
$ yttm decode --help

Usage: yttm decode [OPTIONS]

  Decode ids to text.

Options:
  --model PATH  Path to file with learned model.  [required]
  --ignore_ids  List of indices to ignore for decoding. Example: --ignore_ids=1,2,3
  --help        Show this message and exit.
```







