import pandas as pd
import click
import os


@click.command()
@click.argument('files', nargs=-1, type=click.Path())
def main(files):
    """
    Join multiple pandas DataFrame pickles of NER ground-truth into one big file.
    """

    assert(len(files) > 1)

    gt = list()

    for filename in files[:-1]:

        gt.append(pd.read_pickle(filename))

    gt = pd.concat(gt, axis=0)

    os.makedirs(os.path.dirname(files[-1]), exist_ok=True)

    gt.to_pickle(files[-1])


if __name__ == '__main__':
    main()
