# Running benchmark

* Install [YouTokenToMe](https://github.com/vkcom/youtokentome)
* Install [SentencePiece](https://github.com/google/sentencepiece)
* Compile [fastBPE](https://github.com/glample/fastBPE) and specify path to binary file in variable
 `PATH_TO_FASTBPE` in `speed_test.py`  
* `python speed_test.py`

    **Warning**: this script downloads several GB of data.
    It use Wikipedia monolingual corpora for training and tokenization.
[Here](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) 
 you can find more details about the data. 
