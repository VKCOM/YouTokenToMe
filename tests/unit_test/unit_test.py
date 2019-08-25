import os
import random
import shutil
from pathlib import Path

import youtokentome as yttm

BASE_MODEL_FILE = "artifacts/base_model.yttm"
RENAME_ID_MODEL_FILE = "artifacts/rename_model.yttm"
TRAIN_FILE = "artifacts/random_train_text.txt"
TEST_FILE = "artifacts/random_test_text.txt"

artifacts_generated = False


def generate_artifacts():
    global artifacts_generated
    if not artifacts_generated:
        shutil.rmtree("artifacts", ignore_errors=True)
        Path("artifacts").mkdir()
    else:
        return

    artifacts_generated = True

    n_lines = 10000
    n_characters = 100

    for file_name, alphabet in zip([TRAIN_FILE, TEST_FILE], ["abcd ", "abcde "]):
        with open(file_name, "w") as fout:
            for _ in range(n_lines):
                random_line = "".join(
                    [random.choice(alphabet) for _ in range(n_characters)]
                )
                print(random_line, file=fout)

    cmd = f"yttm bpe --data={TRAIN_FILE} --model={BASE_MODEL_FILE} --vocab_size=16000 --coverage=0.999"
    assert os.system(cmd) == 0

    cmd = f"yttm bpe --data={TRAIN_FILE} --model={RENAME_ID_MODEL_FILE} --vocab_size=16000 --coverage=0.999 --bos_id=29 --eos_id=1148 --unk_id=2922"
    assert os.system(cmd) == 0


def file_starts_with(file_name, pattern):
    with open(file_name, "r") as fin:
        first_line = fin.readline()
        res = first_line.startswith(pattern)
        if not res:
            print("first_line: ", first_line)
        assert res


def test_bos_eos_reverse():
    generate_artifacts()
    cmd = f"yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=1 --bos < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "<BOS>")

    cmd = f"yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=1 --reverse --eos < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "<EOS>")

    cmd = f"yttm encode --model={BASE_MODEL_FILE} --output_type=id --n_threads=1 --bos < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "2")

    cmd = f"yttm encode --model={BASE_MODEL_FILE} --output_type=id --n_threads=1 --reverse --eos < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "3")
    os.remove('log.txt')

def test_interactive_mode():
    generate_artifacts()
    print("interactive helper running id ...")
    cmd = f"python interactor.py | yttm encode --stream --model={BASE_MODEL_FILE} --output_type=id > log.txt"
    assert os.system(cmd) == 0

    print("interactive helper running subword ...")
    cmd = f"python interactor.py | yttm encode --stream --model={BASE_MODEL_FILE} --output_type=subword > log.txt"
    assert os.system(cmd) == 0
    os.remove('log.txt')


def test_inference_speed():
    generate_artifacts()
    print("parallel inference 1 thread  ...")
    cmd = f"time yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=1 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0

    print("parallel inference 2 threads  ...")
    cmd = f"time yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=2 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0

    print("parallel inference 10 threads  ...")
    cmd = f"time yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=10 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0

    print("parallel inference 24 threads  ...")
    cmd = f"time yttm encode --model={BASE_MODEL_FILE} --output_type=subword --n_threads=24 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    os.remove('log.txt')


def test_renaming():
    generate_artifacts()

    cmd = f"yttm encode --model={RENAME_ID_MODEL_FILE} --output_type=id --bos --n_threads=1 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "29")

    cmd = f"yttm encode --model={RENAME_ID_MODEL_FILE} --output_type=id --eos --reverse  --n_threads=1 < {TEST_FILE} > log.txt"
    assert os.system(cmd) == 0
    file_starts_with("log.txt", "1148")
    os.remove('log.txt')


def test_renaming_unknown():
    generate_artifacts()
    with open("local_test.txt", "w") as fout:
        fout.write("z")

    cmd = f"yttm encode --model={RENAME_ID_MODEL_FILE} --output_type=id --reverse  --n_threads=1 < local_test.txt > log.txt"
    assert os.system(cmd) == 0

    file_starts_with("log.txt", "2922")
    os.remove("local_test.txt")
    os.remove('log.txt')
    return


def test_vocab():
    generate_artifacts()
    assert os.system(f"yttm vocab --model {BASE_MODEL_FILE} > /dev/null") == 0
    assert os.system(f"yttm vocab --model {BASE_MODEL_FILE} --verbose > /dev/null") == 0


def test_decode():
    generate_artifacts()
    text_in = " ".join("".join([random.choice("abcd ") for _ in range(50)]).split())

    with open("decode_text_in.txt", "w") as fout:
        fout.write(text_in)
    cmd = f"yttm encode --model {BASE_MODEL_FILE} --output_type=id < decode_text_in.txt > decode_id.txt"
    assert os.system(cmd) == 0
    cmd = f"yttm decode --model {BASE_MODEL_FILE} < decode_id.txt > decode_text_out.txt"
    assert os.system(cmd) == 0

    with open("decode_text_out.txt", "r") as fin:
        text_out = fin.readline()

    os.remove("decode_text_in.txt")
    os.remove("decode_text_out.txt")
    os.remove("decode_id.txt")

    print("input :", text_in + "#")
    print("output:", text_out[:-1] + "#")

    assert text_in == text_out[:-1]


def test_python_api():
    generate_artifacts()
    os.remove(BASE_MODEL_FILE)
    yttm.BPE.train(data=TRAIN_FILE, vocab_size=16000, model=BASE_MODEL_FILE)

    bpe = yttm.BPE(BASE_MODEL_FILE)
    assert bpe.vocab_size() == len(bpe.vocab())
    assert bpe.vocab_size() == len(set(bpe.vocab()))
    vc = bpe.vocab()
    for i, subword in enumerate(vc):
        assert i == bpe.subword_to_id(subword)
        assert subword == bpe.id_to_subword(i)

    text_in = " ".join("".join([random.choice("abcd ") for _ in range(50)]).split())
    ids = bpe.encode(text_in, yttm.OutputType.ID)
    assert text_in == bpe.decode(ids)[0]


def test_stress():
    build_files = ["bpe.cpp", "utils.cpp", "utf8.cpp"]
    files = " ".join(
        ["../../youtokentome/cpp/" + file_name for file_name in build_files]
    )
    files += " stress_test.cpp"

    print("compiling stress test ...")
    cmd = f"g++ {files} -o test -std=c++14 -pthread -D_GLIBCXX_DEBUG -DDETERMINISTIC_QUEUE"

    assert os.system(cmd) == 0
    assert os.system("./test 1000") == 0
    os.remove('test')
    os.remove('remove_it.txt')

