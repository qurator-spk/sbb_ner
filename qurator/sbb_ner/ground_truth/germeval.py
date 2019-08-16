import pandas as pd
import click
import os


def read_gt(files, datasets):
    sentence_number = 200000
    gt_data = list()

    for filename, dataset in zip(files, datasets):
        gt_lines = [l.strip() for l in open(filename)]

        word_number = 0

        for li in gt_lines:

            if li == '':

                if word_number > 0:
                    sentence_number += 1
                    word_number = 0

                continue

            if li.startswith('#'):
                continue

            _, word, tag, _ = li.split()

            tag = tag.upper()
            tag = tag.replace('_', '-')
            tag = tag.replace('.', '-')

            if len(tag) > 5:
                tag = tag[0:5]

            if tag not in {'B-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-ORG', 'I-LOC'}:
                tag = 'O'

            gt_data.append((sentence_number, word_number, word, tag, dataset))

            word_number += 1

    return pd.DataFrame(gt_data, columns=['nsentence', 'nword', 'word', 'tag', 'dataset'])


@click.command()
@click.argument('path-to-germ-eval', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('germ-eval-ground-truth-file', type=click.Path(), required=True, nargs=1)
def main(path_to_germ_eval, germ_eval_ground_truth_file):
    """
    Read germ eval .tsv files from directory <path-to-germ-eval> and
    write the outcome of the data parsing to some pandas DataFrame
    that is stored as pickle in file <germ-eval-ground-truth-file>.
    """

    os.makedirs(os.path.dirname(germ_eval_ground_truth_file), exist_ok=True)

    gt_all = read_gt(['{}/NER-de-dev.tsv'.format(path_to_germ_eval),
                      '{}/NER-de-test.tsv'.format(path_to_germ_eval),
                      '{}/NER-de-train.tsv'.format(path_to_germ_eval)],
                     ['GERM-EVAL-DEV', 'GERM-EVAL-TEST', 'GERM-EVAL-TRAIN'])

    gt_all.to_pickle(germ_eval_ground_truth_file)


if __name__ == '__main__':
    main()
