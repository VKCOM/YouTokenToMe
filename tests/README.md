# Running benchmark

* Install [YouTokenToMe](https://github.com/vkcom/youtokentome)
* Install [SentencePiece](https://github.com/google/sentencepiece)
* Compile [fastBPE](https://github.com/glample/fastBPE) and specify path to binary file in variable
 `PATH_TO_FASTBPE` in `speed_test.py`  
* `python speed_test.py`

    **Warning!** This test requires about **80 GBs** of free space on your disk and can take **several hours** for running.
    It uses Wikipedia monolingual corpora for training and tokenization.
[Here](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) 
 you can find more details about the data.
 
## Docker

Alternatively benchmark can be run using Docker.

```
cd tests
docker build -t yttm/speed_test .
docker run --rm -it yttm/speed_test:latest
```
