import os
import random
from subprocess import run

from utils_for_testing import (
    BASE_MODEL_FILE,
    RENAME_ID_MODEL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    BOS_ID,
    EOS_ID,
    file_starts_with,
    generate_artifacts,
)


def test_bos_eos_reverse():
    generate_artifacts()
    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=subword",
        "--n_threads=1",
        "--bos",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "<BOS>")

    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=subword",
        "--n_threads=1",
        "--reverse",
        "--eos",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "<EOS>")

    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=id",
        "--n_threads=1",
        "--bos",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "2")

    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=id",
        "--n_threads=1",
        "--reverse",
        "--eos",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "3")
    os.remove("log.txt")


def test_interactive_mode():
    generate_artifacts()
    print("interactive helper running id ...")
    cmd = f"python interactor.py | yttm encode --stream --model={BASE_MODEL_FILE} --output_type=id > log.txt"
    assert os.system(cmd) == 0

    print("interactive helper running subword ...")
    cmd = f"python interactor.py | yttm encode --stream --model={BASE_MODEL_FILE} --output_type=subword > log.txt"
    assert os.system(cmd) == 0
    os.remove("log.txt")


def test_multithreading():
    generate_artifacts()
    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=subword",
        "--n_threads=10",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)


def test_renaming():
    generate_artifacts()
    cmd_args = [
        "yttm",
        "encode",
        f"--model={RENAME_ID_MODEL_FILE}",
        "--output_type=id",
        "--bos",
        "--n_threads=1",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "29")

    cmd_args = [
        "yttm",
        "encode",
        f"--model={RENAME_ID_MODEL_FILE}",
        "--output_type=id",
        "--eos",
        "--reverse",
        "--n_threads=1",
    ]
    run(cmd_args, stdin=open(TEST_FILE, "r"), stdout=open("log.txt", "w"), check=True)
    file_starts_with("log.txt", "1148")
    os.remove("log.txt")


def test_renaming_unknown():
    generate_artifacts()
    with open("local_test.txt", "w") as fout:
        fout.write("z")

    cmd_args = [
        "yttm",
        "encode",
        f"--model={RENAME_ID_MODEL_FILE}",
        "--output_type=id",
        "--reverse",
        "--n_threads=1",
    ]
    run(
        cmd_args,
        stdin=open("local_test.txt", "r"),
        stdout=open("log.txt", "w"),
        check=True,
    )

    file_starts_with("log.txt", "2922")
    os.remove("local_test.txt")
    os.remove("log.txt")
    return


def test_vocab():
    generate_artifacts()
    run(["yttm", "vocab", f"--model={BASE_MODEL_FILE}"], check=True)
    run(["yttm", "vocab", f"--model={BASE_MODEL_FILE}", "--verbose"], check=True)


def test_decode():
    generate_artifacts()
    text_in = " ".join("".join([random.choice("abcd ") for _ in range(50)]).split())

    with open("decode_text_in.txt", "w") as fout:
        fout.write(text_in)
    cmd_args = ["yttm", "encode", f"--model={BASE_MODEL_FILE}", "--output_type=id"]
    run(
        cmd_args,
        stdin=open("decode_text_in.txt", "r"),
        stdout=open("decode_id.txt", "w"),
        check=True,
    )

    cmd_args = ["yttm", "decode", f"--model={BASE_MODEL_FILE}"]
    run(
        cmd_args,
        stdin=open("decode_id.txt", "r"),
        stdout=open("decode_text_out.txt", "w"),
        check=True,
    )

    with open("decode_text_out.txt", "r") as fin:
        text_out = fin.readline()

    assert text_in == text_out[:-1]

    cmd_args = [
        "yttm",
        "encode",
        f"--model={BASE_MODEL_FILE}",
        "--output_type=id",
        "--bos",
        "--eos",
    ]
    run(
        cmd_args,
        stdin=open("decode_text_in.txt", "r"),
        stdout=open("decode_id.txt", "w"),
        check=True,
    )

    cmd_args = [
        "yttm",
        "decode",
        f"--model={BASE_MODEL_FILE}",
        f"--ignore_ids={BOS_ID},{EOS_ID}",
    ]
    run(
        cmd_args,
        stdin=open("decode_id.txt", "r"),
        stdout=open("decode_text_out.txt", "w"),
        check=True,
    )

    with open("decode_text_out.txt", "r") as fin:
        text_out = fin.readline()

    assert text_in == text_out[:-1]

    os.remove("decode_text_in.txt")
    os.remove("decode_text_out.txt")
    os.remove("decode_id.txt")
