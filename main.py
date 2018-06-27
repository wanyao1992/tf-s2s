# python main.py --cell_type 'lstm' --attention_type 'luong' --hidden_units 1024 --depth 2 --embedding_size 500  --mode train >~/log/tf-s2s/log.main.train
from __future__ import division
import argparse
import os
import math
import time
import json
import random
import pickle
import numpy as np
import tensorflow as tf

from seq2seq_model import Seq2SeqModel
from trainer import Trainer
from evaluator import Evaluator
from dataset import Dataset
from dict import Dict
import sys

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/demo-train.pt', help='Path to source vocabulary')
    parser.add_argument('--source_vocabulary', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/train.clean.clean.en.json', help='Path to source vocabulary')
    parser.add_argument('--target_vocabulary', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/train.clean.clean.de.json', help='Path to source vocabulary')
    parser.add_argument('--source_train_data', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/train.clean.bpe.en', help='Path to source training data')
    parser.add_argument('--target_train_data', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/train.clean.bpe.de', help='Path to target training data')
    parser.add_argument('--source_valid_data', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/newstest2012.tok.bpe.32000.en', help='Path to source validation data')
    parser.add_argument('--target_valid_data', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/newstest2012.tok.bpe.32000.en', help='Path to target validation data')

    parser.add_argument('--cell_type', required=True, default='lstm', help='RNN cell for encoder and decoder, default: lstm')
    parser.add_argument('--attention_type', required=True, default='bahdanau', help='Attention mechanism: (bahdanau, luong), default: bahdanau')
    parser.add_argument("--hidden_units", type=int, default=1024, help=("Number of hidden units in each layer"))
    parser.add_argument("--depth", type=int, default=2, help=("Number of layers in each encoder and decoder"))
    parser.add_argument("--embedding_size", type=int, default=500, help=("Embedding dimensions of encoder and decoder inputs"))
    # parser.add_argument("--num_encoder_symbols", type=int, default=30000,
    #                     help=("Source vocabulary size"))
    # parser.add_argument("--num_decoder_symbols", type=int, default=30000, help=("Target vocabulary size"))
    parser.add_argument('--use_residual', action="store_true", default=True, help='Use residual connection between layers')
    parser.add_argument('--attn_input_feeding', action="store_true", default=False, help='Use input feeding method in attentional decoder')
    parser.add_argument('--use_dropout', action="store_true", default=True, help='Use dropout in each rnn cell')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout probability for input/output/state units (0.0: no dropout)')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--max_gradient_norm', type=float, default=1.0, help='Clip gradients to this norm')
    parser.add_argument("--batch_size", type=int, default=64, help=("Batch size"))
    parser.add_argument("--max_epochs", type=int, default=2, help=("Maximum # of training epochs"))
    parser.add_argument("--max_load_batches", type=int, default=20, help=("Maximum # of batches to load at one time"))
    parser.add_argument("--max_seq_length", type=int, default=50, help=("Maximum sequence length"))
    parser.add_argument("--display_freq", type=int, default=100, help=("Display training status every this iteration"))
    parser.add_argument("--save_freq", type=int, default=11500, help=("Save model checkpoint every this iteration"))
    parser.add_argument("--valid_freq", type=int, default=1150000, help=("Evaluate model every this iteration: valid_data needed"))
    parser.add_argument('--optimizer', default='adam', help='Optimizer for training: (adadelta, adam, rmsprop)')
    parser.add_argument('--model_dir', default='/media/BACKUP/ghproj_d/tf-seq2seq/model/', help='Path to save model checkpoints')
    parser.add_argument('--model_name', default='translate.ckpt', help='File name used for model checkpoints')
    parser.add_argument('--shuffle_each_epoch', action="store_true", default=True, help='Shuffle training dataset for each epoch')
    parser.add_argument('--sort_by_length', action="store_true", default=True, help='Sort pre-fetched minibatches by their target sequence lengths')
    # parser.add_argument('--use_fp16', action="store_true", default=False, help='Use half precision float16 instead of float32 as dtype')

    # Runtime parameters
    parser.add_argument('--allow_soft_placement', action="store_true", default=True, help='Allow device soft placement')
    parser.add_argument('--log_device_placement', action="store_true", default=False, help='Log placement of ops on devices')

    parser.add_argument('--restore', action="store_true", default=False, help='whether restore from a model file')
    parser.add_argument('--mode', required=False, default='train', help='mode: train/eval')

    # Decoding parameters
    parser.add_argument("--beam_width", type=int, default=12, help='Beam width used in beamsearch')
    parser.add_argument("--decode_batch_size", type=int, default=80, help='Batch size used for decoding')
    parser.add_argument("--max_decode_step", type=int, default=100, help='Maximum time step limit to decode')
    parser.add_argument('--write_n_best', action="store_true", default=False, help='Write n-best list (n=beam_width)')
    parser.add_argument('--model_path', required=False, default=None, help='Path to a specific model checkpoint.')
    # parser.add_argument('--decode_input', required=False, default='data/newstest2012.bpe.de', help= 'Decoding input path')
    parser.add_argument('--decode_output', required=False, default='/media/BACKUP/ghproj_d/tf-seq2seq/pred.txt', help='Decoding output path')
    args = parser.parse_args()
    return args

def create_model(session, args, dicts):
    model = Seq2SeqModel(args, dicts)
    if args.restore:
        print('Reloading model parameters..')
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        model.restore(session, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model

if __name__ == '__main__':
    args = get_args()
    dataset = pickle.load(open(args.data, 'rb'))

    train_data = Dataset(dataset["train"], args.batch_size, eval=False)
    valid_data = Dataset(dataset["valid"], args.batch_size, eval=False)
    test_data = Dataset(dataset["test"], args.batch_size, eval=False)
    dicts = dataset["dicts"]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=args.allow_soft_placement,
                                          log_device_placement=args.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        if args.mode == 'train':
            model = create_model(sess, args, dicts)
            trainer = Trainer(sess, model, train_data, valid_data, args)
            trainer.train()
        elif args.mode == 'test':
            args.restore = True
            model = create_model(sess, args, dicts)
            evaluator = Evaluator(sess, model, test_data, dicts, args)
            evaluator.eval()