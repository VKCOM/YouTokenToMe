# -*- coding: utf-8 -*-
import youtokentome as yttm


def check(text, vocab, output_type=yttm.OutputType.ID):
    TEXT_FILE = "text_file.txt"
    VOCAB_FILE = "vocab_file.txt"
    with open(TEXT_FILE, 'w') as f:
        f.write(text)
    with open(VOCAB_FILE, 'w') as f:
        for word in vocab:
            f.write(word)
            f.write('\n')

    encoder = yttm.WordPiece(VOCAB_FILE)
    return encoder.encode(TEXT_FILE, output_type=output_type)


def test_russian():
    ids = check("привет мир", ["привет", "мир"])
    assert ids == [0, 1]

    ids = check("привет мир", ["при", "##вет", "мир"])
    assert ids == [0, 1, 2]

    ids = check("токенизация это круто", ["ток", "крут", "это", "##за", "##ция", "ция"])
    assert ids == [-1, 2, -1]

    ids = check("токенизация это круто", ["ток", "крут", "это", "##за", "##ени", "##о", "##ция", "ция"])
    assert ids == [0, 4, 3, 6, 2, 1, 5]


def test_english():
    ids = check("self-made", ["self", "made", "-", "##-", "##made"])
    assert ids == [0, 2, 1]

    ids = check("self, made", ["self", "made", ",", "##,", "##made"])
    assert ids == [0, 2, 1]

    ids = check("self  , made", ["self", "made", ",", "##,", "##made"])
    assert ids == [0, 2, 1]


def test_japanese():
    pass


def test_misc():
    ids = check("abcdef", ["a", "##bcdef", "ab", "##c", "##d", "##e", "##f"])
    assert ids == [2, 3, 4, 5, 6]

    ids = check("abcdef abc abcd", ["abcd", "def", "abc"])
    assert ids == [-1, 2, 0]

    ids = check("abc", ["a", "abd"])
    assert ids == [-1]

    ids = check("abc a abc abd", ["a", "abd"])
    assert ids == [-1, 0, -1, 1]

    ids = check("abcdef", ["bcde", "ac", "def", "bc", "bcdef", "##a", "##b", "##c", "##d"])
    assert ids == [-1]
