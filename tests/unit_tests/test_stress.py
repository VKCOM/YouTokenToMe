import os
from subprocess import run


tests_compiled = False

def compile_test():
    global tests_compiled
    if tests_compiled:
        return
    build_files = ["bpe.cpp", "utils.cpp", "utf8.cpp"]
    files = ["../../youtokentome/cpp/" + file_name for file_name in build_files]
    files.append("stress_test.cpp")

    print("compiling stress test ...")

    command = [
        "g++",
        *files,
        "-o",
        "stress",
        "-std=c++11",
        "-pthread",
        "-Og",
        "-D_GLIBCXX_DEBUG",
        "-fno-omit-frame-pointer -fsanitize=address -fsanitize=leak -fsanitize=undefined",
        "-DDETERMINISTIC_QUEUE",
    ]

    command = " ".join(command)
    print("command:", command)
    run(command, check=True, shell=True)
    tests_compiled = True


def test_stress():
    compile_test()
    run(["./stress", "base", "1000"], check=True)


def test_manual():
    compile_test()
    run(["./stress", "manual"], check=True)
    os.remove("remove_it.txt")


def test_parallel():
    compile_test()
    run(["./stress", "parallel", "50"], check=True)
    os.remove("remove_it.txt")
