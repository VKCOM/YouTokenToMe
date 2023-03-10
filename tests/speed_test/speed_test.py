import argparse
import os
from pathlib import Path
from time import time

from tabulate import tabulate
from tokenizers import pre_tokenizers
from tokenizers import Tokenizer as HuggingFaceBPETokenizer
from tokenizers.models import BPE as HuggingFaceBPEModel
from tokenizers.trainers import BpeTrainer as HuggingFaceBPETrainer

MODEL_FILE_NAME = "bpe.model"
MODEL_SUFFIX = ".model"

YOU_TOKEN_TO_ME = "YouTokenToMe"
SENTENCE_PIECE = "SentencePiece"
FAST_BPE = "fastBPE"
HUGGING_FACE_BPE = "Hugging_Face_BPE"

PATH_TO_FASTBPE = "./fastBPE"


class HuggingfaceInterface:
    def train_from_file(self, train_file, vocab_size, model_file, _):
        tokenizer = HuggingFaceBPETokenizer(HuggingFaceBPEModel(unk_token="[UNK]"))
        trainer = HuggingFaceBPETrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=vocab_size)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        tokenizer.train([str(train_file)], trainer)
        tokenizer.save(model_file)

    def encode_file(self, model_path, path_in, path_out, _):
        tokenizer = HuggingFaceBPETokenizer.from_file(model_path)
        with open(path_in) as fin:
            full_text = fin.readlines()
        tokenizer.encode_batch(full_text)


class SentencePieceInterface:
    def train_from_file(self, train_file, vocab_size, model_file, _):
        tmp = model_file.split(".")
        assert len(tmp) == 2
        assert tmp[1] == "model"
        train_command = f"spm_train "
        train_command += f" --input={str(train_file)} "
        train_command += f" --model_prefix={tmp[0]} "
        train_command += f" --vocab_size={vocab_size} "
        train_command += f" --character_coverage=1.0 "
        train_command += f" --model_type=bpe "
        assert os.system(train_command) == 0

    def encode_file(self, model_path, path_in, path_out, _):
        encode_command = f"spm_encode "
        encode_command += f" --model={model_path} "
        encode_command += f" --output_format=piece "
        encode_command += f" < {path_in} > {path_out} "
        assert os.system(encode_command) == 0


class FastBPEInterface:
    def train_from_file(self, file_path, vocab_size, model_file, _):
        train_command = f"{PATH_TO_FASTBPE} learnbpe"
        train_command += f" {vocab_size} {str(file_path)} > {model_file}"
        assert os.system(train_command) == 0

    def encode_file(self, model_path, path_in, path_out, _):
        encode_command = f"{PATH_TO_FASTBPE} applybpe {path_out} {path_in} {model_path}"
        assert os.system(encode_command) == 0


class YouTokenToMeInterface:
    def train_from_file(self, file_path, vocab_size, model_path, n_threads):
        train_command = f"yttm bpe "
        train_command += f" --data={file_path} --model={model_path} "
        train_command += f" --vocab_size={vocab_size}  --n_threads={n_threads} "
        assert os.system(train_command) == 0

    def encode_file(self, model_path, path_in, path_out, n_threads):
        encode_command = "yttm encode "
        encode_command += f" --model={model_path} --output_type=id "
        encode_command += f" --n_threads={n_threads} "
        encode_command += f" < {str(path_in)} > {str(path_out)}"
        assert os.system(encode_command) == 0


def get_bpe(impl_name):
    if impl_name == YOU_TOKEN_TO_ME:
        return YouTokenToMeInterface()
    if impl_name == SENTENCE_PIECE:
        return SentencePieceInterface()
    if impl_name == FAST_BPE:
        return FastBPEInterface()
    if impl_name == HUGGING_FACE_BPE:
        return HuggingfaceInterface()
    assert False


def check_train(algorithm, vocab_size, corpus_path, n_threads):
    bpe = get_bpe(algorithm)
    start_time = time()
    bpe.train_from_file(corpus_path, vocab_size, MODEL_FILE_NAME, n_threads)
    return time() - start_time


def check_inference_file(algorithm, corpus_path, n_threads):
    bpe = get_bpe(algorithm)
    start_time = time()
    bpe.encode_file(MODEL_FILE_NAME, corpus_path, "rm_it.txt", n_threads)
    return time() - start_time


def download_xml2txt():
    if not Path("xml2txt.pl").exists():
        print("downloading xml2txt.pl ...")
        os.system("wget https://www.dropbox.com/s/p3ta9spzfviovk0/xml2txt.pl")


def prepare_data(zip_path, size_mb):
    expected_extension = ".xml.bz2"
    assert zip_path.endswith(expected_extension)
    base_path = Path(zip_path).parent

    unzip_path = base_path / "wiki.xml"
    full_text_path = base_path / "wiki.txt"
    cutted_text_path = base_path / f"wiki_{size_mb}MB.txt"

    if not Path(unzip_path).exists():
        print(f"unziping file {zip_path} ...")
        assert os.system(f"bzip2 -kdc {zip_path} > {unzip_path}") == 0

    if not Path(full_text_path).exists():
        print(f"converting xml to text {unzip_path} ...")
        download_xml2txt()
        preprocess_command = f"perl xml2txt.pl "
        preprocess_command += f" -nomath -notables "
        preprocess_command += f" {unzip_path} {full_text_path}"
        assert os.system(preprocess_command) == 0

    if not Path(cutted_text_path).exists():
        byte_processed = 0
        with open(cutted_text_path, "w") as fout:
            with open(full_text_path, "r") as fin:
                while byte_processed < size_mb * 1_000_000:
                    s = fin.readline()
                    byte_processed += len(s.encode())
                    fout.write(s)

    return cutted_text_path


def speed_test(corpus_path, vocab_size, algorithms, n_threads):
    train_res = {}
    infer_res = {}
    for algorithm in algorithms:
        time_train = check_train(algorithm, vocab_size, corpus_path, n_threads)
        time_infer = check_inference_file(algorithm, corpus_path, n_threads)

        train_res[algorithm] = time_train
        infer_res[algorithm] = time_infer

    return train_res, infer_res


def print_results(cfg, result_name, corpuses, algorithms):
    result_table = [
        ["#" for _ in range(len(corpuses) + 1)] for _ in range(len(algorithms))
    ]
    table_header = ["#"] + [lang for lang in corpuses]
    rev_lang = {lang: i for i, lang in enumerate(table_header)}
    rev_algo = {algo: i for i, algo in enumerate(algorithms)}
    for i, algo_name in enumerate(algorithms):
        result_table[i][0] = algo_name

    for lang, res in cfg.items():
        best = min(res.values())
        for algo in res:
            j = rev_lang[lang]
            i = rev_algo[algo]
            multiplier_str = f"{res[algo]/best:.1f}".rstrip('0').rstrip('.')
            result_table[i][j] = f"{res[algo]:.1f} (x{multiplier_str})"

    table_header[0] = result_name
    column_align = ["left"] + ["center" for _ in corpuses]
    print(tabulate(result_table, table_header, tablefmt="github", colalign=column_align))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument(
        "--corpus_size", type=int, default=100, help="Size of testing corpus in MB"
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        help="list of languages for speed test",
        default="ru",
    )

    return parser.parse_args()


def main(args):
    langs = args.langs if isinstance(args.langs, list) else [args.langs]
    # Hugging Face - limit number of processes
    os.environ["RAYON_RS_NUM_CPUS"] = str(args.n_threads)

    short_to_long_names = {
        "en": "English",
        "ru": "Russian",
        "ja": "Japanese",
        "zh": "Chinese",
    }

    # For adding more languages check out this page https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/
    all_links = {
        "English": "https://www.dropbox.com/s/cnrhd11zdtc1pic/enwiki-20181001-corpus.xml.bz2?dl=1",
        "Russian": "https://www.dropbox.com/s/lpfmyrl7nxn5ugg/ruwiki-20181001-corpus.xml.bz2?dl=1",
        "Japanese": "https://www.dropbox.com/s/wf496hlu512z9kc/jawiki-20140807-corpus.xml.bz2?dl=1",
        "Chinese": "https://www.dropbox.com/s/czhr6s5jwaljeue/zhwiki-20140804-corpus.xml.bz2?dl=1",
    }
    links = {
        short_to_long_names[lang]: all_links[short_to_long_names[lang]]
        for lang in langs
    }

    corpuses = {}
    Path("data").mkdir(exist_ok=True)
    for lang, link in links.items():
        Path(f"data/{lang}").mkdir(exist_ok=True)
        zip_file = f"data/{lang}/wiki.xml.bz2"
        if not Path(zip_file).exists():
            os.system(f"wget -O {zip_file} {link}")
        corpuses[lang] = prepare_data(zip_file, args.corpus_size)

    algorithms = [YOU_TOKEN_TO_ME, HUGGING_FACE_BPE, SENTENCE_PIECE, FAST_BPE]

    global_train = {}
    global_tokenization = {}

    for lang, corpus_path in corpuses.items():
        train_stat, tokenization_stat = speed_test(
            corpus_path, args.vocab_size, algorithms, args.n_threads
        )
        global_train[lang] = train_stat
        global_tokenization[lang] = tokenization_stat

    print_results(global_train, f"Train {args.corpus_size}MB", corpuses, algorithms)
    print_results(global_tokenization, f"Tokenization {args.corpus_size}MB", corpuses, algorithms)


if __name__ == "__main__":
    args = parse_args()
    main(args)
