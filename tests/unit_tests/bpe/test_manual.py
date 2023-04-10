# -*- coding: utf-8 -*-
import os

import youtokentome as yttm


def test_russian():
    train_text = """
        собирать cборник сборище отобранный сборщица 
        """

    test_text = """
        собранный собрание прибор
        """

    TRAIN_DATA_PATH = "train_data.txt"
    MODEL_PATH = "model.yttm"
    with open(TRAIN_DATA_PATH, "w") as fin:
        fin.write(train_text)
    model = yttm.BPE.train(TRAIN_DATA_PATH, MODEL_PATH, 50)
    tokenized_text = model.encode([test_text], output_type=yttm.OutputType.SUBWORD)
    expected_result = [
        ["▁с", "обранный", "▁с", "об", "ран", "и", "е", "▁", "п", "р", "и", "бор"]
    ]
    assert tokenized_text == expected_result
    print(tokenized_text)
    os.remove(TRAIN_DATA_PATH)


def test_english():
    train_text = """
        anachronism
        synchronous  
        chronology
        chronic
        chronophilia
        chronoecological
        chronocoulometry
        """

    test_text = "chronocline synchroscope "

    TRAIN_DATA_PATH = "train_data.txt"
    MODEL_PATH = "model.yttm"
    with open(TRAIN_DATA_PATH, "w") as fin:
        fin.write(train_text)
    model = yttm.BPE.train(TRAIN_DATA_PATH, MODEL_PATH, 200, n_threads=1)
    tokenized_text = model.encode([test_text], output_type=yttm.OutputType.SUBWORD)
    expected_result = [['▁chrono', 'c', 'l', 'i', 'n', 'e', '▁', 'sy', 'n', 'ch', 'r', 'o', 's', 'co', 'p', 'e']]
    assert tokenized_text == expected_result
    print(tokenized_text)
    os.remove(TRAIN_DATA_PATH)


def test_japanese():
    train_text = """
        むかし、 むかし、 ある ところ に
        おじいさん と おばあさん が いました。
        おじいさん が 山（やま） へ 木（き） を きり に いけば、
        おばあさん は 川（かわ） へ せんたく に でかけます。
        「おじいさん、 はよう もどって きなされ。」
        「おばあさん も き を つけて な。」
        まい日（にち） やさしく いい あって でかけます 
    """
    test_text = " おばあさん が  川 で せん "
    TRAIN_DATA_PATH = "train_data.txt"
    MODEL_PATH = "model.yttm"
    with open(TRAIN_DATA_PATH, "w") as fin:
        fin.write(train_text)
    model = yttm.BPE.train(TRAIN_DATA_PATH, MODEL_PATH, 100)
    tokenized_text = model.encode([test_text], output_type=yttm.OutputType.SUBWORD)
    expected_result = [["▁おばあさん", "▁が", "▁", "川", "▁", "で", "▁", "せ", "ん"]]
    assert tokenized_text == expected_result
    print(tokenized_text)
    os.remove(TRAIN_DATA_PATH)
