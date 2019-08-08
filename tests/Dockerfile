FROM ubuntu:18.04

RUN apt update
RUN apt install git -y
RUN apt install python3-pip cmake make g++ vim wget -y
RUN pip3 install tabulate youtokentome

WORKDIR /repos

RUN git clone https://github.com/google/sentencepiece.git
RUN git clone https://github.com/glample/fastBPE

WORKDIR /repos/sentencepiece/build

RUN cmake .. &&  make -j $(nproc) && make install && ldconfig -v

WORKDIR /repos/fastBPE

RUN g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast

WORKDIR /test_dir

COPY ./speed_test.py ./speed_test.py
RUN cp /repos/fastBPE/fast /test_dir/fastBPE

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD python3 speed_test.py 


