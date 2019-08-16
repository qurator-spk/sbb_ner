import pandas as pd
import re
import click
import os


def read_gt(files, datasets):
    sentence_number = 100000
    sentence = ''
    gt_data = list()

    for filename, dataset in zip(files, datasets):
        gt_lines = [l.strip() for l in open(filename) if not l.startswith('<--')]

        word_number = 0

        for l in gt_lines:

            try:
                word, tag = l.split(' ')
            except ValueError:
                word = l.replace(' ', '_')
                tag = 'O'

            tag = tag.upper()

            tag = tag.replace('_', '-')
            tag = tag.replace('.', '-')

            if tag not in {'B-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-ORG', 'I-LOC'}:
                tag = 'O'

            gt_data.append((sentence_number, word_number, word, tag, dataset))

            if re.match(r'.*[.|?|!]$', word) \
               and not re.match(r'[0-9]+[.]$', word) \
               and not re.match(r'.*[0-9]+\s*$', sentence)\
               and not re.match(r'.*\s+[\S]{1,2}$', sentence):

                sentence_number += 1
                sentence = ''
                word_number = 0
            else:
                word_number += 1
                sentence += ' ' + word

    return pd.DataFrame(gt_data, columns=['nsentence', 'nword', 'word', 'tag', 'dataset'])


@click.command()
@click.argument('path-to-ner-corpora', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('ner-ground-truth-file', type=click.Path(), required=True, nargs=1)
def main(path_to_ner_corpora, ner_ground_truth_file):
    """
    Read europeana historic ner ground truth .bio files from directory <path-to-ner-corpora> and
    write the outcome of the data parsing to some pandas DataFrame
    that is stored as pickle in file <ner-ground-truth-file>.
    """

    os.makedirs(os.path.dirname(ner_ground_truth_file), exist_ok=True)

    gt_all = read_gt(['{}/enp_DE.sbb.bio/enp_DE.sbb.bio'.format(path_to_ner_corpora),
                      '{}/enp_DE.onb.bio/enp_DE.onb.bio'.format(path_to_ner_corpora),
                      '{}/enp_DE.lft.bio/enp_DE.lft.bio'.format(path_to_ner_corpora)], ['SBB', 'ONB', 'LFT'])

    gt_all.to_pickle(ner_ground_truth_file)


if __name__ == '__main__':
    main()
