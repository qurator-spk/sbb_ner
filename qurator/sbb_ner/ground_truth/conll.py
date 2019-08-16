import pandas as pd
import click
import codecs
import os


def read_gt(files, datasets):
    sentence_number = 300000
    gt_data = list()

    for filename, dataset in zip(files, datasets):
        gt_lines = [l.strip() for l in codecs.open(filename, 'r', 'latin-1')]

        word_number = 0

        for li in gt_lines:

            if li == '':

                if word_number > 0:

                    sentence_number += 1
                    word_number = 0

                continue

            if li.startswith('-DOCSTART-'):
                continue

            parts = li.split()

            if len(parts) == 5:
                word, _, _, _, tag = li.split()
            else:
                word, _, _, tag = li.split()

            tag = tag.upper()
            tag = tag.replace('_', '-')
            tag = tag.replace('.', '-')

            if tag not in {'B-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-ORG', 'I-LOC'}:
                tag = 'O'

            gt_data.append((sentence_number, word_number, word, tag, dataset))

            word_number += 1

    return pd.DataFrame(gt_data, columns=['nsentence', 'nword', 'word', 'tag', 'dataset'])


@click.command()
@click.argument('path-to-conll', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('conll-ground-truth-file', type=click.Path(), required=True, nargs=1)
def main(path_to_conll, conll_ground_truth_file):
    """
    Read CONLL 2003 ner ground truth files from directory <path-to-conll> and
    write the outcome of the data parsing to some pandas DataFrame
    that is stored as pickle in file <conll-ground-truth-file>.
    """

    os.makedirs(os.path.dirname(conll_ground_truth_file), exist_ok=True)

    gt_all = read_gt(['{}/deu.dev'.format(path_to_conll),
                      '{}/deu.testa'.format(path_to_conll),
                      '{}/deu.testb'.format(path_to_conll),
                      '{}/deu.train'.format(path_to_conll),
                      '{}/eng.testa'.format(path_to_conll),
                      '{}/eng.testb'.format(path_to_conll),
                      '{}/eng.train'.format(path_to_conll)],
                     ['DE-CONLL-DEV', 'DE-CONLL-TESTA', 'DE-CONLL-TESTB', 'DE-CONLL-TRAIN',
                      'EN-CONLL-TESTA', 'EN-CONLL-TESTB', 'EN-CONLL-TRAIN'])

    gt_all.to_pickle(conll_ground_truth_file)


if __name__ == '__main__':
    main()
