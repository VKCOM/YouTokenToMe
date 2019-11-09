import _youtokentome_cython as yttmc
import click


@click.group()
def main():
    pass


@click.command()
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Training data file path.",
)
@click.option(
    "--model", type=click.Path(), required=True, help="Output model file path."
)
@click.option(
    "--vocab_size",
    type=click.INT,
    required=True,
    help="Number of tokens in the final vocabulary.",
)
@click.option(
    "--coverage",
    type=click.FLOAT,
    help="Amount of characters covered by the model.",
    default=1.0,
    show_default=True,
)
@click.option(
    "--n_threads",
    type=click.INT,
    help="Number of threads.",
    default=-1,
    show_default=True,
)
@click.option(
    "--pad_id", type=click.INT, help="Padding token id.", default=0, show_default=True
)
@click.option(
    "--unk_id", type=click.INT, help="Unknown token id.", default=1, show_default=True
)
@click.option(
    "--bos_id",
    type=click.INT,
    help="Begin of sentence token id.",
    default=2,
    show_default=True,
)
@click.option(
    "--eos_id",
    type=click.INT,
    help="End of sentence token id.",
    default=3,
    show_default=True,
)
def bpe(data, model, vocab_size, coverage, n_threads, pad_id, unk_id, bos_id, eos_id):
    """Train BPE model."""
    yttmc.BPE.train(
        data=data,
        model=model,
        vocab_size=vocab_size,
        coverage=coverage,
        n_threads=n_threads,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to file with learned model.",
)
@click.option(
    "--output_type", type=click.STRING, required=True, help="'id' or 'subword'."
)
@click.option(
    "--n_threads",
    type=click.INT,
    help="Number of threads.",
    default=-1,
    show_default=True,
)
@click.option("--bos", is_flag=True, help="Add tab begin of sentence.")
@click.option("--eos", is_flag=True, help="Add tab end of sentence.")
@click.option("--reverse", is_flag=True, help="Reverse output sequence of tokens.")
@click.option(
    "--stream", is_flag=True, help="Process each line before reading the next one."
)
@click.option(
    "--dropout_prob", type=click.FLOAT, default=0, show_default=True, help="Process each line before reading the next one."
)
def encode(model, output_type, n_threads, bos, eos, reverse, stream, dropout_prob):
    """Encode text to ids or subwords."""
    output_type = output_type.lower()
    if output_type != "id" and output_type != "subword":
        raise ValueError(
            'Invalid value for "--output_type": must be equal to "id" or "subword", not "%d".'
            % output_type
        )
    if n_threads < -1 or n_threads == 0:
        raise ValueError(
            'Invalid value for "--n_threads": must be -1 or positive integer, not "%d"'
            % n_threads
        )

    bpe = yttmc.BPE(model, n_threads)
    bpe.encode_cli(output_type, stream, bos, eos, reverse, dropout_prob)


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to file with learned model.",
)
def decode(model):
    """Decode ids to text."""
    bpe = yttmc.BPE(model)
    bpe.decode_cli()


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to file with learned model.",
)
@click.option("--verbose", is_flag=True, help="Add merging rules.")
def vocab(model, verbose):
    """Print list of learned subwords."""
    bpe = yttmc.BPE(model)
    bpe.vocab_cli(verbose)


main.add_command(bpe)
main.add_command(encode)
main.add_command(decode)
main.add_command(vocab)
