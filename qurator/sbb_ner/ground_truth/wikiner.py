import pandas as pd
import click
import os


def read_gt(files, datasets):

    sentence_number = 1000000
    gt_data = list()

    for filename, dataset in zip(files, datasets):

        for li in open(filename, encoding='iso-8859-1'):

            li = li.strip()

            parts = li.split(' ')

            prev_tag = 'O'
            for word_number, pa in enumerate(parts):

                if len(pa) == 0:
                    continue

                word, pos, tag = pa.split('|')

                tag = tag.upper()
                tag = tag.replace('_', '-')
                tag = tag.replace('.', '-')

                if len(tag) > 5:
                    tag = tag[0:5]

                if tag not in {'B-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-ORG', 'I-LOC'}:
                    tag = 'O'

                if tag.startswith('I') and prev_tag == 'O':
                    tag = 'B' + tag[1:]

                prev_tag = tag
                gt_data.append((sentence_number, word_number, word, tag, dataset))

            sentence_number += 1

    return pd.DataFrame(gt_data, columns=['nsentence', 'nword', 'word', 'tag', 'dataset'])


@click.command()
@click.argument('path-to-wikiner', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('wikiner-ground-truth-file', type=click.Path(), required=True, nargs=1)
def main(path_to_wikiner, wikiner_ground_truth_file):
    """
    Read wikiner files from directory <path-to-wikiner> and
    write the outcome of the data parsing to some pandas DataFrame
    that is stored as pickle in file <wikiner-ground-truth-file>.
    """

    os.makedirs(os.path.dirname(wikiner_ground_truth_file), exist_ok=True)

    gt_all = read_gt(['{}/aij-wikiner-de-wp2'.format(path_to_wikiner),
                      '{}/aij-wikiner-de-wp3'.format(path_to_wikiner)],
                     ['WIKINER-WP2', 'WIKINER-WP3'])

    gt_all.to_pickle(wikiner_ground_truth_file)


if __name__ == '__main__':
    main()
