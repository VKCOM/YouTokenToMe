import random
import shutil
from pathlib import Path
from subprocess import run

BASE_MODEL_FILE = "artifacts/base_model.yttm"
RENAME_ID_MODEL_FILE = "artifacts/rename_model.yttm"
TRAIN_FILE = "artifacts/random_train_text.txt"
TEST_FILE = "artifacts/random_test_text.txt"
BOS_ID = 2
EOS_ID = 3

artifacts_generated = False


def generate_artifacts():
    global artifacts_generated
    if not artifacts_generated:
        shutil.rmtree("artifacts", ignore_errors=True)
        Path("artifacts").mkdir()
    else:
        return
    random.seed(19)

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

    cmd_args = [
        "yttm",
        "bpe",
        f"--data={TRAIN_FILE}",
        f"--model={BASE_MODEL_FILE}",
        "--vocab_size=16000",
        "--coverage=0.999",
        f"--bos_id={BOS_ID}",
        f"--eos_id={EOS_ID}",
    ]

    run(cmd_args, check=True)
    cmd_args = [
        "yttm",
        "bpe",
        f"--data={TRAIN_FILE}",
        f"--model={RENAME_ID_MODEL_FILE}",
        "--vocab_size=16000",
        "--coverage=0.999",
        "--bos_id=29",
        "--eos_id=1148",
        "--unk_id=2922",
    ]
    run(cmd_args, check=True)


def file_starts_with(file_name, pattern):
    with open(file_name, "r") as fin:
        first_line = fin.readline()
        res = first_line.startswith(pattern)
        if not res:
            print("first_line: ", first_line)
        assert res
