

## Speed tests

`YouTokenToMe` will be compared with [SentencePiece](https://github.com/google/sentencepiece/)
 and [fastBPE](https://github.com/glample/fastBPE). These two algorithms are considered to be fast.
 
Data from [Wikipedia](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) was used to evaluate algorithm speed. In a similar way to `enwik8` and `enwik9`, the experiments were run on first `10^8` and `10^9` bytes of datasets for English, Russian, Chinese and Japanese.

`vocab_size` was set to the commonly used value `30000`.

In this benchmark, `YouTokenToMe` used 4 threads for training and tokenization. `SentencePiece`
 doesn't support multithreading for **BPE** at all. `fastBPE` doesn't support multithreading for training. 
 For tokenization, it also used 4 threads. 
 
 The results of the experiments are below. The time is measured in seconds.



### Training 100MB

 | |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
SentencePiece|86.5 (x18.8)|82.1 (x17.4)|963.6 (x68.8)|1026.4 (x91.6)
fastBPE|41.0 (x8.9)|36.1 (x7.6)|700.9 (x50.0)|485.1 (x43.3)
YouTokenToMe|**4.6** (x1)|**4.7** (x1)|**14** (x1)|**11.2** (x1)



### Tokenization 100MB
 | |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
SentencePiece|31.1 (x9.7)|51.5 (x10.3)|22.9 (x6.9)|23.2 (x8.0)
fastBPE|10.1 (x3.1)|10.3 (x2.0)|56.3 (x17.0)|55.3 (x19.0)
YouTokenToMe|**3.2** (x1)|**5.0** (x1)|**3.3** (x1)|**2.9** (x1)


### Training 1GB
 | |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
SentencePiece|455.9 (x14.9)|454.5 (x15.5)|3035.7 (x31.7)|5485.7 (x53.0)
fastBPE|293.3 (x9.6)|253.4 (x8.6)|4388.2 (x45.8)|4554.8 (x44.0)
YouTokenToMe|**30.4** (x1)|**29.3** (x1)|**95.7** (x1)|**103.5** (x1)


### Tokenization 1GB 

  | |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
SentencePiece|319.2 (x10.0)|543.1 (x10.9)|228.1 (x7.3)|220.5 (x7.1)
fastBPE|92.5 (x2.9)|116 (x2.3)|444.3 (x14.3)|545.7 (x17.7)
YouTokenToMe|**31.7** (x1)|**49.5** (x1)|**30.9** (x1)|**30.7** (x1)


`YouTokenToMe` performed really well in this benchmark. This is especially noticeable for languages with large alphabets.


## Number of threads

The table below shows the dependence of performance on the number of threads for `YouTokenToMe`.

### Training 1GB
 | YouTokenToMe |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
1 thread |60.5|71.2|236.9|233.0
2 threads|38.0|41.5|137.7|141.7
4 threads|30.4|29.3|95.7|103.5
8 threads|**23.4**|**25.2**|**74.1**|**77.5**
16 threads|25.4|23.5|84.2|85.5

### Tokenization 1GB

 | YouTokenToMe |**Russian**|**English**|**Chinese**|**Japanese**
:-----:|:-----:|:-----:|:-----:|:-----:
1 thread|86.7|135.3|62.4|65.9
2 threads|51.5|81.9|43.8|42.6
4 threads|33.2|52.6|32.4|32.5
8 threads|25.4|38.9|28.9|25.8
16 threads|20.1|32.4|24.9|20.6


Training performance stops increasing significantly after 8 threads. 
So the number of threads for training is always `min(8, n_threads)`. 
In almost any situation, `n_threads=-1` is a good value to use. 
In this case, the number of threads will be determined automatically.


All experiments were run on the following machine:
24-core Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz, 256GB memory, Debian 8.10




