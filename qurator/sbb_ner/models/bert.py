from __future__ import absolute_import, division, print_function
# from inspect import currentframe

import argparse
import logging
import os
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME,  # WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer


from conlleval import evaluate as conll_eval

from tqdm import tqdm, trange

from qurator.sbb_ner.ground_truth.data_processor import NerProcessor, WikipediaNerProcessor

from sklearn.model_selection import GroupKFold

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def model_train(bert_model, max_seq_length, do_lower_case,
                num_train_epochs, train_batch_size, gradient_accumulation_steps,
                learning_rate, weight_decay, loss_scale, warmup_proportion,
                processor, device, n_gpu, fp16, cache_dir, local_rank,
                dry_run, no_cuda, output_dir=None):

    label_map = processor.get_labels()

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    train_dataloader = processor.get_train_examples(train_batch_size, local_rank)

    # Batch sampler divides by batch_size!
    num_train_optimization_steps = int(len(train_dataloader)*num_train_epochs/gradient_accumulation_steps)

    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                         'distributed_{}'.format(local_rank))

    model = BertForTokenClassification.from_pretrained(bert_model, cache_dir=cache_dir, num_labels=len(label_map))

    if fp16:
        model.half()

    model.to(device)

    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion, t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
        warmup_linear = None

    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    logger.info("  Num epochs = %d", num_train_epochs)

    model_config = {"bert_model": bert_model, "do_lower": do_lower_case,
                    "max_seq_length": max_seq_length, "label_map": label_map}

    def save_model(lh):

        if output_dir is None:
            return

        output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(ep))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        torch.save(model_to_save.state_dict(), output_model_file)

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))

        lh = pd.DataFrame(lh, columns=['global_step', 'loss'])

        loss_history_file = os.path.join(output_dir, "loss_ep{}.pkl".format(ep))

        lh.to_pickle(loss_history_file)

    def load_model(epoch):

        if output_dir is None:

            return False

        output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(epoch))

        if not os.path.exists(output_model_file):

            return False

        logger.info("Loading epoch {} from disk...".format(epoch))
        model.load_state_dict(torch.load(output_model_file,
                                         map_location=lambda storage, loc: storage if no_cuda else None))
        return True

    model.train()
    for ep in trange(1, int(num_train_epochs) + 1, desc="Epoch"):

        if dry_run and ep > 1:
            logger.info("Dry run. Stop.")
            break

        if load_model(ep):
            global_step += len(train_dataloader) // gradient_accumulation_steps
            continue

        loss_history = list()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {ep}") as pbar:
        
            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_history.append((global_step, loss.item()))

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")

                if dry_run and len(loss_history) > 2:
                    logger.info("Dry run. Stop.")
                    break

                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = learning_rate * warmup_linear.get_lr(global_step, warmup_proportion)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        save_model(loss_history)

    return model, model_config


def model_eval(batch_size, label_map, processor, device, num_train_epochs=1, output_dir=None, model=None,
               local_rank=-1, no_cuda=False, dry_run=False):

    output_eval_file = None
    if output_dir is not None:
        output_eval_file = os.path.join(output_dir, processor.get_evaluation_file())
        logger.info('Write evaluation results to: {}'.format(output_eval_file))

    dataloader = processor.get_dev_examples(batch_size, local_rank)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataloader))
    logger.info("  Batch size = %d", batch_size)

    results = list()

    output_config_file = None
    if output_dir is not None:
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

    for ep in trange(1, int(num_train_epochs) + 1, desc="Epoch"):

        if dry_run and ep > 1:
            logger.info("Dry run. Stop.")
            break

        if output_config_file is not None:
            # Load a trained model and config that you have fine-tuned
            output_model_file = os.path.join(output_dir, "pytorch_model_ep{}.bin".format(ep))

            if not os.path.exists(output_model_file):
                logger.info("Stopping at epoch {} since model file is missing.".format(ep))
                break

            config = BertConfig(output_config_file)
            model = BertForTokenClassification(config, num_labels=len(label_map))
            model.load_state_dict(torch.load(output_model_file,
                                             map_location=lambda storage, loc: storage if no_cuda else None))
            model.to(device)

        if model is None:
            raise ValueError('Model required for evaluation.')

        model.eval()

        y_pred, y_true = model_predict_compare(dataloader, device, label_map, model, dry_run)

        lines = ['empty ' + 'XXX ' + v + ' ' + p for yt, yp in zip(y_true, y_pred) for v, p in zip(yt, yp)]

        res = conll_eval(lines)

        # print(res)

        evals = \
            pd.concat([pd.DataFrame.from_dict(res['overall']['evals'], orient='index', columns=['ALL']),
                       pd.DataFrame.from_dict(res['slots']['LOC']['evals'], orient='index', columns=['LOC']),
                       pd.DataFrame.from_dict(res['slots']['PER']['evals'], orient='index', columns=['PER']),
                       pd.DataFrame.from_dict(res['slots']['ORG']['evals'], orient='index', columns=['ORG']),
                       ], axis=1).T

        stats = \
            pd.concat(
                [pd.DataFrame.from_dict(res['overall']['stats'], orient='index', columns=['ALL']),
                 pd.DataFrame.from_dict(res['slots']['LOC']['stats'], orient='index', columns=['LOC']),
                 pd.DataFrame.from_dict(res['slots']['PER']['stats'], orient='index', columns=['PER']),
                 pd.DataFrame.from_dict(res['slots']['ORG']['stats'], orient='index', columns=['ORG'])],
                axis=1, sort=True).T

        evals['epoch'] = ep
        stats['epoch'] = ep

        results.append(pd.concat([evals.reset_index().set_index(['index', 'epoch']),
                                  stats.reset_index().set_index(['index', 'epoch'])], axis=1))

        if output_eval_file is not None:
            pd.concat(results).to_pickle(output_eval_file)

    results = pd.concat(results)
    print(results)

    return results


def model_predict_compare(dataloader, device, label_map, model, dry_run=False):

    y_true = []
    y_pred = []
    covered = set()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] != "X":
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
                else:
                    temp_1.pop()
                    temp_2.pop()
                    y_true.append(temp_1)
                    y_pred.append(temp_2)

                    covered = covered.union(set(temp_1))
                    break

        if dry_run:

            if 'I-LOC' not in covered:
                continue
            if 'I-ORG' not in covered:
                continue
            if 'I-PER' not in covered:
                continue

            break
    return y_pred, y_true


def model_predict(dataloader, device, label_map, model):

    y_pred = []
    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, mask in enumerate(input_mask):
            temp_2 = []
            for j, m in enumerate(mask):
                if j == 0:  # skip first token since its [CLS]
                    continue
                if m:
                    temp_2.append(label_map[logits[i][j]])
                else:
                    temp_2.pop()  # skip last token since its [SEP]
                    y_pred.append(temp_2)
                    break
            else:
                y_pred.append(temp_2)

    return y_pred


def get_device(local_rank=-1, no_cuda=False):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device, n_gpu


def main():

    parser = get_arg_parser()

    args = parser.parse_args()

    do_eval = len(args.dev_sets) > 0 and not args.do_cross_validation
    do_train = len(args.train_sets) > 0 and not args.do_cross_validation

    device, n_gpu = get_device(args.local_rank, args.no_cuda)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not do_train and not do_eval and not args.do_cross_validation:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    processors = {"ner": NerProcessor, "wikipedia-ner": WikipediaNerProcessor}

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    if args.do_cross_validation:

        cross_val_result_file = "cross_validation_results.pkl"

        cross_val_result_file = os.path.join(args.output_dir, cross_val_result_file)

        sets = set(args.train_sets.split('|')) if args.train_sets is not None else set()

        gt = pd.read_pickle(args.gt_file)

        gt = gt.loc[gt.dataset.isin(sets)]

        k_fold = GroupKFold(n_splits=args.n_splits)

        eval_results = list()

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        for ep in range(1, int(args.num_train_epochs) + 1):

            for sp, (train, test) in enumerate(k_fold.split(X=gt, groups=gt.nsentence)):

                tr = gt.iloc[train].copy()
                te = gt.iloc[test].copy()

                tr['dataset'] = 'TRAIN'
                te['dataset'] = 'TEST'

                gt_tmp = pd.concat([tr, te])

                processor = \
                    processors[task_name](train_sets='TRAIN', dev_sets='TEST', test_sets='TEST',
                                          gt=gt_tmp, max_seq_length=args.max_seq_length,
                                          tokenizer=tokenizer, data_epochs=args.num_data_epochs,
                                          epoch_size=args.epoch_size)

                model, model_config = \
                    model_train(bert_model=args.bert_model, max_seq_length=args.max_seq_length,
                                do_lower_case=args.do_lower_case, num_train_epochs=ep,
                                train_batch_size=args.train_batch_size,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                loss_scale=args.loss_scale, warmup_proportion=args.warmup_proportion,
                                processor=processor, device=device, n_gpu=n_gpu, fp16=args.fp16,
                                cache_dir=args.cache_dir, local_rank=args.local_rank, dry_run=args.dry_run,
                                no_cuda=args.no_cuda)

                label_map = {v: k for k, v in model_config['label_map'].items()}

                eval_result =\
                    model_eval(model=model, label_map=label_map, processor=processor, device=device,
                               batch_size=args.eval_batch_size, local_rank=args.local_rank,
                               no_cuda=args.no_cuda, dry_run=args.dry_run).reset_index()

                eval_result['split'] = sp
                eval_result['epoch'] = ep
                eval_results.append(eval_result)

                del model  # release CUDA memory

            pd.concat(eval_results).to_pickle(cross_val_result_file)

    if do_train:

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        processor = \
            processors[task_name](train_sets=args.train_sets, dev_sets=args.dev_sets, test_sets=args.test_sets,
                                  gt_file=args.gt_file, max_seq_length=args.max_seq_length,
                                  tokenizer=tokenizer, data_epochs=args.num_data_epochs,
                                  epoch_size=args.epoch_size)

        model_train(bert_model=args.bert_model, output_dir=args.output_dir, max_seq_length=args.max_seq_length,
                    do_lower_case=args.do_lower_case, num_train_epochs=args.num_train_epochs,
                    train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay, loss_scale=args.loss_scale,
                    warmup_proportion=args.warmup_proportion, processor=processor, device=device, n_gpu=n_gpu,
                    fp16=args.fp16, cache_dir=args.cache_dir, local_rank=args.local_rank, dry_run=args.dry_run,
                    no_cuda=args.no_cuda)

    if do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        model_config = json.load(open(os.path.join(args.output_dir, "model_config.json"), "r"))

        label_to_id = model_config['label_map']

        label_map = {v: k for k, v in model_config['label_map'].items()}

        tokenizer = BertTokenizer.from_pretrained(model_config['bert_model'],
                                                  do_lower_case=model_config['do_lower'])

        processor = \
            processors[task_name](train_sets=None, dev_sets=args.dev_sets, test_sets=args.test_sets,
                                  gt_file=args.gt_file, max_seq_length=model_config['max_seq_length'],
                                  tokenizer=tokenizer, data_epochs=args.num_data_epochs,
                                  epoch_size=args.epoch_size, label_map=label_to_id)

        model_eval(label_map=label_map, processor=processor, device=device, num_train_epochs=args.num_train_epochs,
                   output_dir=args.output_dir, batch_size=args.eval_batch_size, local_rank=args.local_rank,
                   no_cuda=args.no_cuda, dry_run=args.dry_run)


def get_arg_parser():

    parser = argparse.ArgumentParser()


    parser.add_argument("--gt_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The pickle file that contains all NER ground truth as pandas DataFrame."
                             " Required columns: ['nsentence', 'nword', 'word', 'tag', 'dataset]."
                             " The selection of training, test and dev set is performed on the 'dataset' column.")

    parser.add_argument("--train_sets",
                        default='',
                        type=str,
                        required=False,
                        help="Specifiy one or more tags from the dataset column in order to mark samples"
                             " that belong to the training set. Example: 'GERM-EVAL-TRAIN|DE-CONLL-TRAIN'. ")

    parser.add_argument("--dev_sets",
                        default='',
                        type=str,
                        required=False,
                        help="Specifiy one or more tags from the dataset column in order to mark samples"
                             " that belong to the dev set. Example: 'GERM-EVAL-DEV|DE-CONLL-TESTA'. ")

    parser.add_argument("--test_sets",
                        default='',
                        type=str,
                        required=False,
                        help="Specifiy one or more tags from the dataset column in order to mark samples"
                             " that belong to the test set. Example: 'GERM-EVAL-TEST|DE-CONLL-TESTB'. ")

    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform/evaluate.")

    parser.add_argument("--num_data_epochs",
                        default=1.0,
                        type=float,
                        help="Re-cycle data after num_data_epochs.")

    parser.add_argument("--epoch_size",
                        default=10000,
                        type=float,
                        help="Size of one epoch.")

    parser.add_argument("--do_cross_validation",
                        action='store_true',
                        help="Do cross-validation.")

    parser.add_argument("--n_splits",
                        default=5,
                        type=int,
                        help="Number of folds in cross_validation.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--dry_run",
                        action='store_true',
                        help="Test mode.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    return parser


if __name__ == "__main__":
    main()
