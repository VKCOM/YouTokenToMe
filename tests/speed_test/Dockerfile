FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	cmake \
	make \
	g++ \
	wget \
	bzip2 \
	perl && \
	pip3 install -r requirements.txt && \
    pip3 install youtokentome

WORKDIR /repos

RUN git clone https://github.com/google/sentencepiece.git && \
    git clone https://github.com/glample/fastBPE

WORKDIR /repos/sentencepiece/build

RUN cmake .. &&  make -j $(nproc) && make install && ldconfig -v

WORKDIR /repos/fastBPE

RUN g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast

WORKDIR /workspace

RUN cp /repos/fastBPE/fast /workspace/fastBPE
RUN wget -O bert-base-cased.txt https://huggingface.co/bert-base-cased/resolve/main/vocab.txt

COPY ./bpe.py ./bpe.py
COPY ./wordpiece.py ./wordpiece.py

# use comma to separate langs, e.g.: "--langs", "en", "ru", "zh", "ja"
CMD ["python", "bpe.py", "--langs", "ru", "--corpus_size", "10", "--vocab_size", "30000"]

CMD ["python", "bpe.py", "--langs", "ru", "--corpus_size", "10", "--vocab", "bert-base-cased.txt"]