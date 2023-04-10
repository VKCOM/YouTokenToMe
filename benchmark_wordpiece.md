## WordPiece Speed tests

`YouTokenToMe` will be compared with:
* [Hugging Face](https://github.com/huggingface/tokenizers)
* [Keras](https://github.com/keras-team/keras-nlp)
* [Tensorflow](https://github.com/tensorflow/text)
* [Torch](https://github.com/pytorch/text)

These algorithms are considered to be fast.

Data from [Wikipedia](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) was used to evaluate algorithm speed. In a similar way to `enwik8` and `enwik9`, the experiments were run on first `10^8` and `10^9` bytes of datasets for English, Russian, Chinese and Japanese.

Used vocabulary: [bert-base-cased](https://huggingface.co/bert-base-cased).

In this benchmark, `YouTokenToMe` used 4 threads for training and tokenization.
 
Source code for benchmark can be found [here](tests/speed_test/wordpiece.py).
The results of the experiments are below. The time is measured in seconds.

All experiments were run on the following machine: TODO

### Tokenization 100MB
TODO: TABLE

### Tokenization 1GB 
TODO: TABLE

`YouTokenToMe` performed really well in this benchmark. This is especially noticeable for languages with large alphabets.

## Number of threads

The table below shows the dependence of performance on the number of threads for `YouTokenToMe`.

### Tokenization 1GB
TODO: TABLE


TODO: CONCLUSION ON THREADS
