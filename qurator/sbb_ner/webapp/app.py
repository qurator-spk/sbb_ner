import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
import html
import json
import torch
from somajo import Tokenizer, SentenceSplitter

from qurator.sbb_ner.models.bert import get_device, model_predict
from qurator.sbb_ner.ground_truth.data_processor import NerProcessor, convert_examples_to_features
from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


class NERPredictor:

    def __init__(self, model_dir, batch_size, epoch, max_seq_length=128, local_rank=-1, no_cuda=False):

        self._batch_size = batch_size
        self._local_rank = local_rank
        self._max_seq_length = max_seq_length

        self._device, self._n_gpu = get_device(no_cuda=no_cuda)

        self._model_config = json.load(open(os.path.join(model_dir, "model_config.json"), "r"))

        self._label_to_id = self._model_config['label_map']

        self._label_map = {v: k for k, v in self._model_config['label_map'].items()}

        self._bert_tokenizer = \
            BertTokenizer.from_pretrained(model_dir,
                                          do_lower_case=self._model_config['do_lower'])

        output_config_file = os.path.join(model_dir, CONFIG_NAME)

        output_model_file = os.path.join(model_dir, "pytorch_model_ep{}.bin".format(epoch))

        config = BertConfig(output_config_file)

        self._model = BertForTokenClassification(config, num_labels=len(self._label_map))
        self._model.load_state_dict(torch.load(output_model_file,
                                               map_location=lambda storage, loc: storage if no_cuda else None))
        self._model.to(self._device)
        self._model.eval()

        return

    def classify_text(self, sentences):

        examples = NerProcessor.create_examples(sentences, 'test')

        features = [fe for ex in examples for fe in
                    convert_examples_to_features(ex, self._label_to_id, self._max_seq_length, self._bert_tokenizer)]

        data_loader = NerProcessor.make_data_loader(None, self._batch_size, self._local_rank, self._label_to_id,
                                                    self._max_seq_length, self._bert_tokenizer, features=features,
                                                    sequential=True)

        prediction_tmp = model_predict(data_loader, self._device, self._label_map, self._model)

        assert len(prediction_tmp) == len(features)

        prediction = []
        prev_guid = None
        for fe, pr in zip(features, prediction_tmp):
            # longer sentences might have been processed in several steps
            # therefore we have to glue them together. This can be done on the basis of the guid.

            if prev_guid != fe.guid:
                prediction.append((fe.tokens[1:-1], pr))
            else:
                prediction[-1] = (prediction[-1][0] + fe.tokens[1:-1], prediction[-1][1] + pr)

            prev_guid = fe.guid

        try:
            assert len(sentences) == len(prediction)
        except AssertionError:
            print('Sentences:\n')
            print(sentences)
            print('\n\nPrediciton:\n')
            print(prediction)

        return prediction


class NERTokenizer:

    def __init__(self):

        self._word_tokenizer = Tokenizer(split_camel_case=True, token_classes=False, extra_info=False)

        self._sentence_splitter = SentenceSplitter()

    def parse_text(self, text):
        tokens = self._word_tokenizer.tokenize_paragraph(text)

        sentences_tokenized = self._sentence_splitter.split(tokens)

        sentences = []
        for sen in sentences_tokenized:

            sen = [tok.replace(" ", "") for tok in sen]

            if len(sen) == 0:
                continue

            sentences.append((sen, []))

        return sentences


class PredictorStore:

    def __init__(self):

        self._predictor = None
        self._model_id = None

    def get(self, model_id):

        if model_id is not None:
            model = next((m for m in app.config['MODELS'] if m['id'] == int(model_id)))
        else:
            model = next((m for m in app.config['MODELS'] if m['default']))

        if self._model_id != model['id']:

            self._predictor = NERPredictor(model_dir=model['model_dir'],
                                           epoch=model['epoch'],
                                           batch_size=app.config['BATCH_SIZE'],
                                           no_cuda=False if not os.environ.get('USE_CUDA') else
                                           os.environ.get('USE_CUDA').lower() == 'false')
            self._model_id = model['id']

        return self._predictor


predictor_store = PredictorStore()

tokenizer = NERTokenizer()


@app.route('/')
def entry():
    return redirect("/index.html", code=302)


@app.route('/models')
def get_models():
    return jsonify(app.config['MODELS'])


@app.route('/tokenized', methods=['GET', 'POST'])
def tokenized():

    raw_text = request.json['text']

    sentences = tokenizer.parse_text(raw_text)

    result = [(sen, i) for i, (sen, _) in enumerate(sentences)]

    return jsonify(result)


@app.route('/ner-bert-tokens', methods=['GET', 'POST'])
@app.route('/ner-bert-tokens/<model_id>', methods=['GET', 'POST'])
def ner_bert_tokens(model_id=None):

    raw_text = request.json['text']

    sentences = tokenizer.parse_text(raw_text)

    prediction = predictor_store.get(model_id).classify_text(sentences)

    output = []

    for tokens, word_predictions in prediction:

        output_sentence = []

        for token, word_pred in zip(tokens, word_predictions):

            output_sentence.append({'token': html.escape(token), 'prediction': word_pred})

        output.append(output_sentence)

    return jsonify(output)


@app.route('/ner', methods=['GET', 'POST'])
@app.route('/ner/<model_id>', methods=['GET', 'POST'])
def ner(model_id=None):

    raw_text = request.json['text']

    sentences = tokenizer.parse_text(raw_text)

    prediction = predictor_store.get(model_id).classify_text(sentences)

    output = []

    for (tokens, word_predictions),  (input_sentence, _) in zip(prediction, sentences):

        original_text = "".join(input_sentence)
        original_word_positions = \
            [pos for positions in [[idx] * len(word) for idx, word in enumerate(input_sentence)] for pos in positions]

        word = ''
        last_prediction = 'O'
        output_sentence = []

        for pos, (token, word_pred) in enumerate(zip(tokens, word_predictions)):

            if not token.startswith('##') and token != '[UNK]':
                if len(word) > 0:
                    output_sentence.append({'word': word, 'prediction': last_prediction})

                word = ''

            if token == '[UNK]':

                orig_pos = len("".join([pred['word'] for pred in output_sentence]) + word)

                if orig_pos > 0 and original_word_positions[orig_pos-1] != original_word_positions[orig_pos]:
                    output_sentence.append({'word': word, 'prediction': last_prediction})
                    word = ''

                word += original_text[orig_pos]

                continue

            token = token[2:] if token.startswith('##') else token

            word += token

            if word_pred != 'X':
                last_prediction = word_pred

        if len(word) > 0:
            output_sentence.append({'word': word, 'prediction': last_prediction})

        output.append(output_sentence)

    for output_sentence, (input_sentence, _) in zip(output, sentences):

        try:
            assert "".join([pred['word'] for pred in output_sentence]) == "".join(input_sentence)
        except AssertionError:
            logger.warning('Input and output different!!! \n\n\nInput: {}\n\nOutput: {}\n'.
                           format("".join(input_sentence).replace(" ", ""),
                                  "".join([pred['word'] for pred in output_sentence])))

    return jsonify(output)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
