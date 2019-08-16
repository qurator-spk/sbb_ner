from __future__ import absolute_import, division, print_function

import os
import json

import numpy as np
import pandas as pd

import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tokens = tokens


class WikipediaDataset(Dataset):
    """
    """

    def __init__(self, set_file, gt_file, data_epochs, epoch_size,
                 label_map, tokenizer, max_seq_length,
                 queue_size=1000, no_entity_fraction=0.0, seed=23,
                 min_sen_len=10, min_article_len=20):

        self._set_file = set_file
        self._subset = pd.read_pickle(set_file)
        self._gt_file = gt_file
        self._data_epochs = data_epochs
        self._epoch_size = epoch_size
        self._label_map = label_map
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._queue_size = queue_size
        self._no_entity_fraction = no_entity_fraction
        self._seed = seed
        self._min_sen_len = min_sen_len
        self._min_article_len = min_article_len

        self._queue = None
        self._data_sequence = None
        self._counter = None
        # noinspection PyUnresolvedReferences
        self._random_state = np.random.RandomState(seed=self._seed)

        self._reset()

        return

    def _next_sample_should_have_entities(self):

        if self._no_entity_fraction <= 0.0:
            return True

        return int(self._counter) % int(1.0 / self._no_entity_fraction) != 0

    def __getitem__(self, index):

        del index
        if self._counter > self._data_epochs * self._epoch_size:
            self._reset()

        while True:

            # get next random sentence
            sen_words, sen_tags = self._queue_next()

            if len(sen_words) < self._min_sen_len:  # Skip all sentences that are to short.
                continue

            if self._has_entities(sen_tags):

                if not self._next_sample_should_have_entities():  # Skip sample if next sample is supposed to
                    # be a no-entity sample
                    continue
            else:
                if self._next_sample_should_have_entities():  # Skip sample if next sample is supposed to be a entity
                    # sample
                    continue
            break

        sample = InputExample(guid="%s-%s" % (self._set_file, self._counter),
                              text_a=sen_words, text_b=None, label=sen_tags)

        features = convert_examples_to_features(sample, self._label_map, self._max_seq_length, self._tokenizer)

        self._counter += 1

        return torch.tensor(features.input_ids, dtype=torch.long), \
               torch.tensor(features.input_mask, dtype=torch.long), \
               torch.tensor(features.segment_ids, dtype=torch.long), \
               torch.tensor(features.label_id, dtype=torch.long)

    def __len__(self):

        return int(self._epoch_size)

    def _reset(self):

        # print('================= WikipediaDataset:_reset ====================== ')

        self._queue = list()
        self._data_sequence = self._sequence()
        self._counter = 0
        # noinspection PyUnresolvedReferences
        # self._random_state = np.random.RandomState(seed=self._seed)

        for _ in range(0, self._queue_size):
            self._queue.append(list())

    def _sequence(self):

        while True:

            for row in pd.read_csv(self._gt_file, chunksize=1, sep=';'):

                page_id = row.page_id.iloc[0]
                text = row.text.iloc[0]
                tags = row.tags.iloc[0]

                if page_id not in self._subset.index:
                    continue

                sentences = [(sen_text, sen_tag) for sen_text, sen_tag in zip(json.loads(text), json.loads(tags))]

                if len(sentences) < self._min_article_len:  # Skip very short articles.
                    continue

                print(page_id)

                yield sentences

    def _queue_next(self):

        nqueue = self._random_state.randint(len(self._queue))

        while len(self._queue[nqueue]) <= 0:
            self._queue[nqueue] = next(self._data_sequence)

        return self._queue[nqueue].pop()

    @staticmethod
    def _has_entities(sen_tags):

        for t in sen_tags:

            if t != 'O':
                return True

        return False


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_evaluation_file(self):
        raise NotImplementedError()


class WikipediaNerProcessor(DataProcessor):

    def __init__(self, train_sets, dev_sets, test_sets, gt_file, max_seq_length, tokenizer,
                 data_epochs, epoch_size, **kwargs):
        del kwargs

        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._train_set_file = train_sets
        self._dev_set_file = dev_sets
        self._test_set_file = test_sets
        self._gt_file = gt_file
        self._data_epochs = data_epochs
        self._epoch_size = epoch_size

    def get_train_examples(self, batch_size, local_rank):
        """See base class."""

        return self._make_data_loader(self._train_set_file, batch_size, local_rank)

    def get_dev_examples(self, batch_size, local_rank):
        """See base class."""

        return self._make_data_loader(self._dev_set_file, batch_size, local_rank)

    def get_labels(self):
        """See base class."""

        labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "X", "[CLS]", "[SEP]"]

        return {label: i for i, label in enumerate(labels)}

    def get_evaluation_file(self):
        dev_set_name = os.path.splitext(os.path.basename(self._dev_set_file))[0]

        return "eval_results-{}.pkl".format(dev_set_name)

    def _make_data_loader(self, set_file, batch_size, local_rank):
        del local_rank

        data = WikipediaDataset(set_file=set_file, gt_file=self._gt_file,
                                data_epochs=self._data_epochs, epoch_size=self._epoch_size,
                                label_map=self.get_labels(), tokenizer=self._tokenizer,
                                max_seq_length=self._max_seq_length)

        sampler = SequentialSampler(data)

        return DataLoader(data, sampler=sampler, batch_size=batch_size)


class NerProcessor(DataProcessor):

    def __init__(self, train_sets, dev_sets, test_sets, max_seq_length, tokenizer,
                 label_map=None, gt=None, gt_file=None, **kwargs):

        del kwargs

        self._max_seg_length = max_seq_length
        self._tokenizer = tokenizer
        self._train_sets = set(train_sets.split('|')) if train_sets is not None else set()
        self._dev_sets = set(dev_sets.split('|')) if dev_sets is not None else set()
        self._test_sets = set(test_sets.split('|')) if test_sets is not None else set()

        self._gt = gt

        if self._gt is None:
            self._gt = pd.read_pickle(gt_file)

        self._label_map = label_map

        print('TRAIN SETS: ', train_sets)
        print('DEV SETS: ', dev_sets)
        print('TEST SETS: ', test_sets)

    def get_train_examples(self, batch_size, local_rank):
        """See base class."""

        return self.make_data_loader(
                            self.create_examples(self._read_lines(self._train_sets), "train"), batch_size, local_rank,
                            self.get_labels(), self._max_seg_length, self._tokenizer)

    def get_dev_examples(self, batch_size, local_rank):
        """See base class."""
        return self.make_data_loader(
                        self.create_examples(self._read_lines(self._dev_sets), "dev"), batch_size, local_rank,
                        self.get_labels(), self._max_seg_length, self._tokenizer)

    def get_labels(self):
        """See base class."""

        if self._label_map is not None:
            return self._label_map

        gt = self._gt
        gt = gt.loc[gt.dataset.isin(self._train_sets.union(self._dev_sets).union(self._test_sets))]

        labels = sorted(gt.tag.unique().tolist()) + ["X", "[CLS]", "[SEP]"]

        self._label_map = {label: i for i, label in enumerate(labels, 1)}

        self._label_map['UNK'] = 0

        return self._label_map

    def get_evaluation_file(self):

        return "eval_results-{}.pkl".format("-".join(sorted(self._dev_sets)))

    @staticmethod
    def create_examples(lines, set_type):

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label

            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    @staticmethod
    def make_data_loader(examples, batch_size, local_rank, label_map, max_seq_length, tokenizer, features=None,
                         sequential=False):

        if features is None:
            features = [convert_examples_to_features(ex, label_map, max_seq_length, tokenizer)
                        for ex in examples]

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if local_rank == -1:
            if sequential:
                train_sampler = SequentialSampler(data)
            else:
                train_sampler = RandomSampler(data)
        else:
            if sequential:
                train_sampler = SequentialSampler(data)
            else:
                train_sampler = DistributedSampler(data)

        return DataLoader(data, sampler=train_sampler, batch_size=batch_size)

    def _read_lines(self, sets):

        gt = self._gt
        gt = gt.loc[gt.dataset.isin(sets)]

        data = list()
        for i, sent in gt.groupby('nsentence'):

            sent = sent.sort_values('nword', ascending=True)

            data.append((sent.word.tolist(), sent.tag.tolist()))

        return data


def convert_examples_to_features(example, label_map, max_seq_length, tokenizer):
    """
    :param example: instance of InputExample
    :param label_map:
    :param max_seq_length:
    :param tokenizer:
    :return:
    """

    words = example.text_a
    word_labels = example.label
    tokens = []
    labels = []

    for i, word in enumerate(words):

        token = tokenizer.tokenize(word)
        tokens.extend(token)

        label_1 = word_labels[i] if i < len(word_labels) else 'O'

        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    n_tokens = []
    segment_ids = []
    label_ids = []
    n_tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        n_tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    n_tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(n_tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # if ex_index < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % example.guid)
    #     logger.info("tokens: %s" % " ".join(
    #         [str(x) for x in tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("label: %s (id = %d)" % (example.label, label_ids))

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids,
                         tokens=n_tokens)
