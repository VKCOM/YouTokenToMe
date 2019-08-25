import random
import sys
from time import sleep

n = 10

for i in range(n):
    print("".join([random.choice("abcd ") for _ in range(100)]), flush=True)
    n_iter = 0
    while True:
        n_iter += 1
        sleep(0.01)
        with open("log.txt", "r") as fin:
            content = fin.readlines()
            tt = len(content)
            assert tt <= i + 1
            if tt == i + 1:
                print("good!", file=sys.stderr)
                break

        assert n_iter < 100
