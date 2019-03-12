import argparse
# import csv
import logging
import os
# import random
# import sys
import pandas as pd
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

NUM_LABELS = 2

def main():
    parser = initialize_parser() 
    args = parser.parse_args() # get command line arguments

    device, n_gpu = get_device_and_n_gpu(args)
    train_batch_size, processor, num_labels, label_list, tokenizer = initialize_support_values(args)
    
    processor.check_data_exists(args.data_dir, args.skip_train)

    model = get_model(args.cache_dir, args.local_rank, args.bert_model, num_labels, device, n_gpu)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if not args.skip_train: 
        train_examples = processor.get_train_examples(args.data_dir, args.num_train_examples) # TODO merge with get_train_dataloader() ?

        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        optimizer = get_optimizer(model, args.learning_rate, args.warmup_proportion, num_train_optimization_steps)
        
        train_dataloader = get_train_dataloader(train_examples, tokenizer, label_list, args, num_train_optimization_steps) 
        tr_loss, nb_tr_steps, global_step = train_model(model, optimizer, train_dataloader, args, device, n_gpu)

        save_model(model, args, num_labels)
    
    model = load_model(args, num_labels, device)

    if (args.local_rank == -1 or torch.distributed.get_rank() == 0): # and args.do_eval:
        eval_dataloader = get_eval_dataloader(processor, args, tokenizer)
        result = eval_model(model, device, eval_dataloader, args.skip_train, tr_loss, nb_tr_steps, global_step)
        record_result(args, result)

def train_model(model, optimizer, train_dataloader, args, device, n_gpu):
    """ Trains the model on each example in the example set. """
    model.train()
    global_step = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1 
    return tr_loss, nb_tr_steps, global_step

def eval_model(model, device, eval_dataloader, skip_train, tr_loss, nb_tr_steps, global_step):
    """ Determines the accuracy of the model's predictions for each example in the test set. 
     Returns as a dictionary with keys 'eval_loss', 'eval_accuracy', 'global_step', 'loss' """
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss/nb_tr_steps if not skip_train else None
    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'global_step': global_step,
        'loss': loss
    }
    return result

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def get_train_dataloader(train_examples, tokenizer, label_list, args, num_train_optimization_steps):
    """ Convert training examples to a DataLoader  """
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    log_training(len(train_examples), args.train_batch_size, num_train_optimization_steps)
    train_data = get_dataset(train_features)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    return DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

def get_eval_dataloader(processor, args, tokenizer):
    """ parses test examples and prepares them into a DataLoader """

    eval_examples = processor.get_test_examples(args.data_dir, args.num_test_examples)
    eval_features = convert_examples_to_features(eval_examples, processor.get_labels(), args.max_seq_length, tokenizer)

    log_evaluating(len(eval_examples), args.eval_batch_size)

    eval_dataset = get_dataset(eval_features)
    eval_sampler = SequentialSampler(eval_dataset)
    return DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
class LeftRightProcessor:
    def get_labels(self):
        "classification labels for the dataset"
        return ["left", "right"]

    def check_data_exists(self, data_dir, skip_train):
        cant_find_train_data = (not skip_train) and (not os.path.exists(data_dir+"/train.csv"))
        if cant_find_train_data: raise ValueError("could not find {}/train.csv".format(data_dir))
        cant_find_test_data = not os.path.exists(data_dir+"/test.csv")
        if cant_find_test_data: raise ValueError("could not find {}/test.csv".format(data_dir))


    def get_train_examples(self, data_dir, num_train_examples):
        examples = []
        train_data = pd.read_csv(data_dir+"/train.csv", lineterminator="\n")
        train_data = train_data.sample(n=num_train_examples)
        for _, row in train_data.iterrows(): 
            guid = row.id 
            text = row.text
            label = row.label
            examples.append(InputExample(guid, text, label=label))
        return examples

    def get_test_examples(self, data_dir, num_test_examples):
        examples = []
        test_data = pd.read_csv(data_dir+"/test.csv", lineterminator="\n")
        test_data = test_data.sample(n=num_test_examples)
        for _, row in test_data.iterrows():
            guid = row.id
            text = row.text
            label = row.label
            examples.append(InputExample(guid, text, label=label))
        return examples

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
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
        self.text = text
        self.label = label
    def __repr__(self):
        return ("id = %d ; label = %s ; text = %s") % (self.guid, self.label, self.text)

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """ convert each example in a list to an InputFeatures """
    labels_to_int = {label:i for i,label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length - 2 :
             tokens = tokens[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = labels_to_int[example.label]
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id
            )
        )
    return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

# TODO find better way of initializing values
def initialize_support_values(args): 
    """ Adjusts batch size for gradient accumulation steps, creates output directories as needed,
     defines the processor, number of labels, the list of label names, and the tokenizer. """
    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps # TODO figure out why/if this is needed

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "eval_results")):
        os.makedirs(os.path.join(args.output_dir, "eval_results"))


    processor = LeftRightProcessor()
    num_labels = NUM_LABELS
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=(not args.cased))

    return train_batch_size, processor, num_labels, label_list, tokenizer

def get_model(cache_dir, local_rank, bert_model, num_labels, device, n_gpu):
    "Returns a model with the given specifications"

    cache_path = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(local_rank))

    model = BertForSequenceClassification.from_pretrained(
        bert_model,
        cache_dir=cache_path,
        num_labels = num_labels
    )
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model

def get_optimizer(model, learning_rate, warmup_proportion, num_train_optimization_steps):
    "Sets the given specifications for the optimizer then returns the optimizer"

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_proportion,
        t_total=num_train_optimization_steps
    )
    return optimizer

def get_dataset(features):
    """ Converts a set of features to a TensorDataset """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

def save_model(model, args, num_labels):
    """ Saves the model to the given output directory """
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(output_config_file)
    model = BertForSequenceClassification(config, num_labels=num_labels) # TODO can skip this if trained?
    model.load_state_dict(torch.load(output_model_file))

def load_model(args, num_labels, device):
    """ Loads the model from the given directories and puts it on the correct computation device  """
    if not args.skip_train:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)
    return model
    
def get_device_and_n_gpu(args):
    """ Determines where to do calculations and how many GPUs the system has """
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    return device, n_gpu

def initialize_parser():
    """ Defines the structure for the argparser """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .csv files (or other data files) for the task."
    )
    parser.add_argument(
        "--bert_model", default="bert-base-uncased", type=str, required=False,
        help="Bert pre-trained model selected in the list: bert-base-uncased (default), "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese."
    )
    parser.add_argument(
        "--output_dir",
        default="/tmp/classifier_output",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written. (default: /tmp/classifier_output)"
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
                "Sequences longer than this will be truncated, and sequences shorter \n"
                "than this will be padded."
    )
    parser.add_argument(
        "--skip_train",
        action='store_true',
        help="Whether to skip training."
    )
    parser.add_argument(
        "--cased",
        action='store_true',
        help="Set this flag if you are using a cased model."
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
                "E.g., 0.1 = 10%% of training."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3"
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=50,
        help="The number of rows from the train data to use"
    )
    parser.add_argument(
        "--num_test_examples",
        type=int,
        default=50,
        help="The number of rows from the train data to use"
    )
    return parser

## LOGGING:
def log_training(num_examples, train_batch_size, num_train_optimization_steps):
    """ logs the beginning of training """
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

def log_evaluating(num_eval_examples, eval_batch_size):
    """ Logs the beginning of the evaluation. """ 
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", num_eval_examples)
    logger.info("  Batch size = %d", eval_batch_size)

def record_result(args, result):
    """ logs and writes the evalutated stats of the model """
    current_time = time.strftime("%m-%d_%I-%M%p")
    output_eval_file = os.path.join(args.output_dir,"eval_results", "result-{}.txt".format(current_time))
    with open(output_eval_file, "w") as writer:
        writer.write("train_examples = %d\n" % (args.num_train_examples,))
        writer.write("test_examples = %d\n\n" % (args.num_test_examples,))
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S',
    level = logging.INFO
)
logger = logging.getLogger(__name__)
## END OF LOGGING

if __name__ == "__main__":
    main()
