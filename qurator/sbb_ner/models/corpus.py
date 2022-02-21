import re
import pandas as pd
from tqdm import tqdm as tqdm
import click
import codecs
import os
import sqlite3

from qurator.utils.parallel import run as prun


class ChunkTask:

    selection = None

    def __init__(self, chunk, min_line_len):

        self._chunk = chunk
        self._min_line_len = min_line_len

    def __call__(self, *args, **kwargs):

        return ChunkTask.reformat_chunk(self._chunk, self._min_line_len)

    @staticmethod
    def reformat_chunk(chunk, min_line_len):
        """
        Process a chunk of documents.

        :param chunk: pandas DataFrame that contains one document per row.
        :param min_line_len: Break the document text up in lines that have this minimum length.
        :return: One big text where the documents are separated by an empty line.
        """

        text = ''

        for i, r in chunk.iterrows():

            if type(r.text) != str:
                continue

            ppn = r.ppn if str(r.ppn).startswith('PPN') else 'PPN' + r.ppn

            filename = str(r['file name'])

            if not ChunkTask.selection.loc[(ppn, filename)].selected.iloc[0]:
                continue

            for se in sentence_split(str(r.text), min_line_len):

                text += se

            text += '\n\n'

        return text

    @staticmethod
    def initialize(selection_file):

        ChunkTask.selection = \
            pd.read_pickle(selection_file).\
                reset_index().\
                set_index(['ppn', 'filename']).\
                sort_index()


def get_csv_chunks(alto_csv_file, chunksize):

    for ch in tqdm(pd.read_csv(alto_csv_file, chunksize=chunksize)):

        yield ch


def get_sqlite_chunks(alto_sqlite_file, chunksize):

    yield pd.DataFrame()

    with sqlite3.connect(alto_sqlite_file) as conn:

        conn.execute('pragma journal_mode=wal')

        total = int(conn.execute('select count(*) from text;').fetchone()[0] / chunksize)

        for ch in tqdm(pd.read_sql('select * from text', conn, chunksize=chunksize), total=total):

            yield ch


def get_chunk_tasks(chunks, min_len_len):

    for chunk in chunks:

        if len(chunk) == 0:
            continue

        yield ChunkTask(chunk, min_len_len)


def sentence_split(s, min_len):
    """
    Reformat text of an entire document such that each line has at least length min_len
    :param s: str
    :param min_len: minimum line length
    :return: reformatted text
    """

    parts = s.split(' ')

    se = ''
    for p in parts:

        se += ' ' + p

        if len(se) > min_len and len(p) > 2 and re.match(r'.*([^0-9])[.]$', p):
            yield se + '\n'
            se = ''

    yield se + '\n'


@click.command()
@click.argument('fulltext-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('selection-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('corpus-file', type=click.Path(), required=True, nargs=1)
@click.option('--chunksize', default=10**4, help="Process the corpus in chunks of <chunksize>. default:10**4")
@click.option('--processes', default=6, help="Number of parallel processes. default: 6")
@click.option('--min-line-len', default=80, help="Lower bound of line length in output file. default:80")
def collect(fulltext_file, selection_file, corpus_file, chunksize, processes, min_line_len):
    """
    Reads the fulltext from a CSV or SQLITE3 file (see also altotool) and write it to one big text file.

    FULLTEXT_FILE: The CSV or SQLITE3 file to read from.

    SELECTION_FILE: Consider only a subset of all pages that is defined by the DataFrame
    that is stored in <selection_file>.

    CORPUS_FILE: The output file that can be used by bert-pregenerate-trainingdata.
    """
    os.makedirs(os.path.dirname(corpus_file), exist_ok=True)

    print('Open {}.'.format(corpus_file))
    corpus_fh = codecs.open(corpus_file, 'w+', 'utf-8')
    corpus_fh.write(u'\ufeff')

    if fulltext_file.endswith('.csv'):
        chunks = get_csv_chunks(fulltext_file, chunksize)
    elif fulltext_file.endswith('.sqlite3'):
        chunks = get_sqlite_chunks(fulltext_file, chunksize)
    else:
        raise RuntimeError('Unsupported input file format.')

    for text in prun(get_chunk_tasks(chunks, min_line_len), processes=processes, initializer=ChunkTask.initialize,
                     initargs=(selection_file,)):

        corpus_fh.write(text)

    corpus_fh.close()

    return


if __name__ == '__main__':
    main()
