# Running benchmark

**Warning!** This test requires about **20 GBs** of free space on your disk and can take **about one hour** for running.
    It uses Wikipedia monolingual corpora for training and tokenization.
[Here](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) 
 you can find more details about the data.

## Recommended approach

Benchmark can be run using Docker.
Substitute `PATH_TO_DOWNLOADED_DATA` with absolute path to the directory where 
wiki dumps will be downloaded.

```bash
cd tests/speed_test
docker build -t yttm/speed_test .
docker run --rm -v PATH_TO_DOWNLOADED_DATA:/workspace/data -it yttm/speed_test:latest
```

## Alternative approach

## BPE benchmark

* Install [YouTokenToMe](https://github.com/vkcom/youtokentome)
* Install [Hugging Face Tokenizer](https://github.com/huggingface/tokenizers)
* Install [SentencePiece](https://github.com/google/sentencepiece)
* Compile [fastBPE](https://github.com/glample/fastBPE) and specify path to binary file in variable
 `PATH_TO_FASTBPE` in `bpe.py`  
* `python bpe.py`

## WordPiece benchmark

* Install [YouTokenToMe](https://github.com/vkcom/youtokentome)
* Install [Hugging Face Tokenizer](https://github.com/huggingface/tokenizers)
* Install [Keras](https://github.com/keras-team/keras-nlp)
* Install [Tensorflow](https://github.com/tensorflow/text)
* Install [Torch](https://github.com/pytorch/text)
* `python wordpiece.py`
