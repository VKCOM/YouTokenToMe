import os
from subprocess import run


def compile_test():
    build_files = ["bpe.cpp", "utils.cpp", "utf8.cpp"]
    files = ["../../youtokentome/cpp/" + file_name for file_name in build_files]
    files.append("stress_test.cpp")

    print("compiling stress test ...")

    command = [
        "g++",
        *files,
        "-o",
        "test",
        "-std=c++14",
        "-pthread",
        "-D_GLIBCXX_DEBUG",
        "-DDETERMINISTIC_QUEUE",
    ]

    command = " ".join(command)
    print("command:", command)
    run(command, check=True, shell=True)


def test_stress():
    compile_test()
    run(["./test", "base", "1000"], check=True)
    os.remove("test")


def test_manual():
    compile_test()
    run(["./test", "manual"], check=True)
    os.remove("test")
    os.remove("remove_it.txt")


def test_parallel():
    compile_test()
    run(["./test", "parallel", "50"], check=True)
    os.remove("test")
    os.remove("remove_it.txt")
