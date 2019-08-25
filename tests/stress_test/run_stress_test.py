import os

build_files = ["bpe.cpp", "utils.cpp", "utf8.cpp"]
files = " ".join(["../../youtokentome/cpp/" + file_name for file_name in build_files])
files += " stress_test.cpp"

print("compiling stress test ...")
cmd = f"g++ {files} -o test -std=c++14 -pthread -D_GLIBCXX_DEBUG -DDETERMINISTIC_QUEUE"

assert os.system(cmd) == 0
assert os.system("./test 1000") == 0
