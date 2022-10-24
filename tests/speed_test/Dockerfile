FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	cmake \
	make \
	g++ \
	wget && \
    pip3 install tabulate youtokentome tokenizers

WORKDIR /repos

RUN git clone https://github.com/google/sentencepiece.git && \
    git clone https://github.com/glample/fastBPE

WORKDIR /repos/sentencepiece/build

RUN cmake .. &&  make -j $(nproc) && make install && ldconfig -v

WORKDIR /repos/fastBPE

RUN g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast

WORKDIR /workspace

COPY ./speed_test.py ./speed_test.py
RUN cp /repos/fastBPE/fast /workspace/fastBPE

# CMD ["python", "speed_test.py", "--langs", "en", "ru", "zh", "ja", "--corpus_size", "100", "--vocab_size", "30000"]
CMD ["python", "speed_test.py", "--langs", "ru", "--corpus_size", "10", "--vocab_size", "30000"]
