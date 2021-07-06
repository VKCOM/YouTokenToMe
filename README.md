# Fork of YouTokenToMe library

You must install this forked library with the following command:
```
pip uninstall youtokentome
python setup.py install
```

For the rest of the YouTokenToMe documentation please refer to the original repo: https://github.com/VKCOM/YouTokenToMe

This fork is using for tokenizing source code data. The main and only change is that BPE algorithm is not restricted to merge tokens between words separated by spaces, but restricted merges only between lines of code.

You can think that the better way for pre-tokenization is not pre-tokenizing source code by `\n`, but using programming language parser and pre-tokenize code to programming language tokens. The main purpose for such "line" restriction is to reduce a number of tokens for the same amount of source code.

Here is an example of tokenization, where `|` is a token separator:

```
def main(|*args, **kwargs):
    |for i in range(|arg|s[1]|):
        |print|(f"|Some |number|: {|i|}")
```
