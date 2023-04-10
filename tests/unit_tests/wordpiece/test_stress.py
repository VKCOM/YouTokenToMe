import os
from subprocess import run


tests_compiled = False

def compile_test():
    global tests_compiled
    if tests_compiled:
        return
    build_files = ["wordpiece.cpp", "utils.cpp", "utf8.cpp"]
    files = ["../../../youtokentome/cpp/" + file_name for file_name in build_files]
    files.append("stress_test.cpp")

    print("compiling wordpiece stress test ...")

    command = [
        "g++",
        *files,
        "-o",
        "wordpiece_stress",
        "-std=c++11",
        "-pthread",
        "-Og",
        "-D_GLIBCXX_DEBUG",
        "-fno-omit-frame-pointer -fsanitize=address -fsanitize=leak -fsanitize=undefined",
    ]

    command = " ".join(command)
    print("command:", command)
    run(command, check=True, shell=True)
    tests_compiled = True


def test_small():
    compile_test()
    run(["./wordpiece_stress", "small"], check=True)


def test_manual():
    compile_test()
    run(["./wordpiece_stress", "large"], check=True)


def test_parallel():
    compile_test()
    run(["./wordpiece_stress", "parallel"], check=True)
