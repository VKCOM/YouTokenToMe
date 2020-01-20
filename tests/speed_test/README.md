# Running benchmark

* Install [YouTokenToMe](https://github.com/vkcom/youtokentome)
* Install [SentencePiece](https://github.com/google/sentencepiece)
* Install [Hugging Face Tokenizer](https://github.com/huggingface/tokenizers)
* Compile [fastBPE](https://github.com/glample/fastBPE) and specify path to binary file in variable
 `PATH_TO_FASTBPE` in `speed_test.py`  
* `python speed_test.py`

    **Warning!** This test requires about **20 GBs** of free space on your disk and can take **about one hour** for running.
    It uses Wikipedia monolingual corpora for training and tokenization.
[Here](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) 
 you can find more details about the data.
 
## Docker

Alternatively benchmark can be run using Docker.
Substitute `PATH_TO_DOWNLOADED_DATA` with absolute path to the directory where 
wiki dumps will be downloaded.

```
cd tests/speed_test
docker build -t yttm/speed_test .
docker run --rm -v PATH_TO_DOWNLOADED_DATA:/workspace/data -it yttm/speed_test:latest
```
