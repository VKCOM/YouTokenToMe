# Fork of YouTokenToMe library

You can install this fork with the following command:
```
pip uninstall youtokentome
python setup.py install
```
Or from our private PyPI on Space `https://packages.jetbrains.team/pypi/p/ccrm/full-line/simple`

For the rest of the YouTokenToMe documentation please refer to the original repo: https://github.com/VKCOM/YouTokenToMe

This fork is using for tokenizing source code data. The main and only change is that BPE algorithm is not restricted to merge tokens between words separated by spaces, but restricted merges only between lines of code.

It may seems that using programming language parser and pre-tokenize code to programming language tokens is the better way of pre-tokenization :). It probably is, but the main goal of our pre-tokenization by `\n` is reducing number of tokens in context for the same amount of source code.

Here is an example of tokenization, where `|` is a token separator:

```
|def main(|*args, **kwargs):|
|    |for i in range(|arg|s[1]|):|
|        |print|(f"|Some |number|: {|i|}")|
```
