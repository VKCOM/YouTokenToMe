## Speed tests

`YouTokenToMe` will be compared with [Hugging Face](https://github.com/huggingface/tokenizers), [SentencePiece](https://github.com/google/sentencepiece/)
 and [fastBPE](https://github.com/glample/fastBPE). These three algorithms are considered to be fast.
 
Data from [Wikipedia](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) was used to evaluate algorithm speed. In a similar way to `enwik8` and `enwik9`, the experiments were run on first `10^8` and `10^9` bytes of datasets for English, Russian, Chinese and Japanese.

`vocab_size` was set to the commonly used value `30000`.

In this benchmark, `YouTokenToMe` used 4 threads for training and tokenization. `SentencePiece`
 doesn't support multithreading for **BPE** at all. `fastBPE` doesn't support multithreading for training. 
 For tokenization, it also used 4 threads. 
 
Source code for benchmark can be found [here](tests/speed_test/speed_test.py).
The results of the experiments are below. The time is measured in seconds.

All experiments were run on the following machine:
36-core Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz, disabled HyperThreading, 256GB RAM, Ubuntu 22.04.

### Training 100MB
|                  |    English     |    Russian     |     Chinese     |    Japanese     |
|------------------|:--------------:|:--------------:|:---------------:|:---------------:|
| YouTokenToMe     |    4.2 (x1)    |    4.5 (x1)    |    11.5 (x1)    |    11.1 (x1)    |
| Hugging_Face_BPE |  10.3 (x2.4)   |  13.1 (x2.9)   |   47.3 (x4.1)   |   61.6 (x5.5)   |
| SentencePiece    |  76.2 (x17.9)  |  85.3 (x19.1)  |  657.4 (x57.2)  |  587.9 (x52.9)  |
| fastBPE          |  32.6 (x7.7)   |  35.0 (x7.9)   |   482.6 (x42)   |  384.2 (x34.5)  |

### Tokenization 100MB
|                  |    English    |    Russian    |    Chinese    |   Japanese    |
|------------------|:-------------:|:-------------:|:-------------:|:-------------:|
| YouTokenToMe     |   4.4 (x1)    |   3.1 (x1)    |   2.9 (x1)    |   2.8 (x1)    |
| Hugging_Face_BPE |  12.0 (x2.7)  |  8.4 (x2.7)   |  10.4 (x3.6)  |  9.4 (x3.4)   |
| SentencePiece    |  43.4 (x9.8)  |  26.6 (x8.6)  |  19.9 (x6.9)  |  20.1 (x7.2)  |
| fastBPE          |  9.5 (x2.1)   |  9.4 (x3.1)   |  23.3 (x8.1)  |  24.8 (x8.9)  |

### Training 1GB
|                  |     English     |     Russian     |     Chinese      |     Japanese     |
|------------------|:---------------:|:---------------:|:----------------:|:----------------:|
| YouTokenToMe     |    25.4 (x1)    |    24.8 (x1)    |    75.4 (x1)     |    106.7 (x1)    |
| Hugging_Face_BPE |   97.7 (x3.8)   |  103.3 (x4.2)   |   516.4 (x6.8)   |  1981.3 (x18.6)  |
| SentencePiece    |  344.1 (x13.5)  |  355.8 (x14.3)  |  1446.5 (x19.2)  |  3908.7 (x36.6)  |
| fastBPE          |  191.4 (x7.5)   |  201.3 (x8.1)   |   3392.0 (x45)   |  3239.5 (x30.4)  |

### Tokenization 1GB 
|                  |    English     |    Russian     |    Chinese     |    Japanese    |
|------------------|:--------------:|:--------------:|:--------------:|:--------------:|
| YouTokenToMe     |   47.6 (x1)    |   28.1 (x1)    |   34.3 (x1)    |   28.2 (x1)    |
| Hugging_Face_BPE |  126.9 (x2.7)  |  96.8 (x3.4)   |   139.4 (x4)   |  117.9 (x4.2)  |
| SentencePiece    |  438.4 (x9.2)  |  264.6 (x9.4)  |  213.5 (x6.2)  |  201.0 (x7.1)  |
| fastBPE          |  88.0 (x1.8)   |   83.2 (x3)    |   240.1 (x7)   |  257.3 (x9.1)  |

`YouTokenToMe` performed really well in this benchmark. This is especially noticeable for languages with large alphabets.

## Number of threads

The table below shows the dependence of performance on the number of threads for `YouTokenToMe`.

### Training 1GB
|            | English | Russian | Chinese | Japanese |
|------------|:-------:|:-------:|:-------:|:--------:|
| 1 thread   |  62.1   |  58.1   |  140.3  |  187.5   |
| 2 threads  |  36.6   |  35.2   |  83.7   |  110.2   |
| 4 threads  |  25.2   |  25.9   |  58.5   |   75.3   |
| 8 threads  |  20.6   |  20.5   |  48.5   |   57.7   |
| 16 threads |  19.8   |  21.8   |  46.9   |   62.8   |

### Tokenization 1GB
|            | English | Russian | Chinese | Japanese |
|------------|:-------:|:-------:|:-------:|:--------:|
| 1 thread   |  116.1  |  74.4   |  49.3   |   57.0   |
| 2 threads  |  68.7   |  46.2   |  38.2   |   39.0   |
| 4 threads  |  43.5   |  32.1   |  32.1   |   30.7   |
| 8 threads  |  30.1   |  23.5   |  33.0   |   21.2   |
| 16 threads |  30.6   |  19.0   |  27.2   |   22.5   |


Training performance stops increasing significantly after 8 threads. 
So the number of threads for training is always `min(8, n_threads)`. 
In almost any situation, `n_threads=-1` is a good value to use. 
In this case, the number of threads will be determined automatically.
