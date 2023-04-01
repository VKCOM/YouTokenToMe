import argparse
import os
from pathlib import Path
from time import time

import keras_nlp
import tensorflow
from tabulate import tabulate
from tensorflow_text import BertTokenizer as TensorflowBertTokenizer
from tokenizers import BertWordPieceTokenizer as HuggingFaceBertTokenizer
from torchtext.transforms import BERTTokenizer as TorchBertTokenizer


YOU_TOKEN_TO_ME = "YouTokenToMe"
HUGGING_FACE = 'Hugging Face'
KERAS = 'Keras'
TENSORFLOW = 'TensorFlow'
TORCH = 'Torch'

ALGORITHMS = [YOU_TOKEN_TO_ME, HUGGING_FACE, KERAS, TENSORFLOW, TORCH]
LOWER_CASE = False


def collect_to_file(out_file, ids):
    if out_file is not None:
        with open(out_file, 'w') as f:
            for i in ids:
                f.write(f'{i} ')

def run_tensorflow(text_file, vocab_file, n_threads, out_file):
    text = ""
    with open(text_file, 'r') as f:
        text = f.read()
    vocab_list = []
    with open(vocab_file, 'r') as f:
        for word in f:
            vocab_list.append(word)
    lookup_table = tensorflow.lookup.StaticVocabularyTable(
        tensorflow.lookup.KeyValueTensorInitializer(
            keys=vocab_list,
            key_dtype=tensorflow.string,
            values=tensorflow.range(
                tensorflow.size(vocab_list, out_type=tensorflow.int64), dtype=tensorflow.int64),
            value_dtype=tensorflow.int64
        ),
        num_oov_buckets=1
    )
    tokenizer = TensorflowBertTokenizer(lookup_table, token_out_type=tensorflow.int64, lower_case=LOWER_CASE)
    ids = tokenizer.tokenize(text).numpy().tolist()
    assert len(ids) > 0
    collect_to_file(out_file, ids)
    return len(ids)


def run_hugging_face(text_file, vocab_file, n_threads, out_file):
    with open(text_file, 'r') as f:
        text = f.read()
    tokenizer = HuggingFaceBertTokenizer(vocab_file, lowercase=LOWER_CASE)
    ids = tokenizer.encode(text).ids
    assert len(ids) > 0
    collect_to_file(out_file, ids)
    return len(ids)


def run_torch(text_file, vocab_file, n_threads, out_file):
    with open(text_file, 'r') as f:
        text = f.read()
    tokenizer = TorchBertTokenizer(vocab_file, do_lower_case=LOWER_CASE)
    ids = tokenizer(text)
    assert len(ids) > 0
    collect_to_file(out_file, ids)
    return len(ids)


def run_keras(text_file, vocab_file, n_threads, out_file):
    with open(text_file, 'r') as f:
        text = f.read()
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab_file, lowercase=LOWER_CASE)
    ids = tokenizer.tokenize(text).numpy().tolist()
    assert len(ids) > 0
    collect_to_file(out_file, ids)
    return len(ids)


def run_you_token_to_me(text_file, vocab_file, n_threads, out_file):
    assert(LOWER_CASE == False)
    out_file = out_file if out_file is not None else ""
    rc = 0 # TODO
    assert rc == 0
    return rc


def get_wordpiece(impl_name):
    if impl_name == YOU_TOKEN_TO_ME:
        return run_you_token_to_me
    elif impl_name == HUGGING_FACE:
        return run_hugging_face
    elif impl_name == KERAS:
        return run_keras
    elif impl_name == TENSORFLOW:
        return run_tensorflow
    elif impl_name == TORCH:
        return run_torch
    assert False


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


def check_inference_file(algorithm, text_file, vocab_file, n_threads, out_file):
    wordpiece = get_wordpiece(algorithm)
    start_time = time()
    res = wordpiece(text_file, vocab_file, n_threads, out_file)
    elapsed = time() - start_time
    print(f"Runner returned: {res}")
    return elapsed


def speed_test(text_file: str, vocab_file: str, algorithms, n_threads: int, collect: bool,):
    result = {}
    for algorithm in algorithms:
        print(f'Running {algorithm}')
        out_file = f"result_{algorithm}.txt" if collect else None
        time_infer = check_inference_file(algorithm, text_file, vocab_file, n_threads, out_file)
        print(f'{algorithm} finished in {time_infer:.1f} sec')
        result[algorithm] = time_infer

    return result


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

    parser.add_argument(
        "--vocab", type=str, required=True, help="path to vocab file"
    )
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument(
        "--corpus_size", type=int, default=10, help="Size of testing corpus in MB"
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        help="list of languages for speed test",
        default="en",
    )
    parser.add_argument("--collect", action="store_true")

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

    global_tokenization = {}

    for lang, corpus_path in corpuses.items():
        tokenization_stat = speed_test(corpus_path, args.vocab, ALGORITHMS, args.n_threads, args.collect)
        global_tokenization[lang] = tokenization_stat

    print_results(global_tokenization, f"Tokenization {args.corpus_size}MB", corpuses, ALGORITHMS)


if __name__ == "__main__":
    args = parse_args()
    main(args)