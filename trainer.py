# python train.py --cell_type 'lstm' --attention_type 'luong' --hidden_units 1024 --depth 2 --embedding_size 500 --num_encoder_symbols 30000 --num_decoder_symbols 30000
from __future__ import division
import argparse
import os
import math
import time
import random
import numpy as np
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
import sys

class Trainer():
    def __init__(self, sess, model, train_data, eval_data, args):
        self.sess = sess
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        # self.evaluator = lib.Evaluator(model, metrics, dicts, opt)
        self.args = args

    def train(self):
        # Create a log writer object
        log_writer = tf.summary.FileWriter(self.args.model_dir, graph=self.sess.graph)
        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print('Training..')
        for epoch_idx in range(self.args.max_epochs):
            # for source_seq, target_seq in train_data:
            for i in range(len(self.train_data)):
                batch = self.train_data[i]
                source, source_len, target, target_len = batch

                tvars = tf.trainable_variables()
                tvars_vals = self.sess.run(tvars)

                step_loss, summary = self.model.train(self.sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                 decoder_inputs=target, decoder_inputs_length=target_len)
                print('step_loss: ')
                print(step_loss)
                loss += float(step_loss) / self.args.display_freq
                words_seen += float(np.sum(source_len + target_len))
                sents_seen += float(source.shape[0])  # batch_size

                if self.model.global_step.eval() % self.args.display_freq == 0:
                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / self.args.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print('Epoch ', self.model.global_epoch_step.eval(), 'Step ', self.model.global_step.eval(), \
                          'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time ', step_time, \
                          '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec))

                    loss, words_seen, sents_seen = 0, 0, 0
                    start_time = time.time()
                    # Record training summary for the current batch
                    log_writer.add_summary(summary, self.model.global_step.eval())

                # Save the model checkpoint
                if self.model.global_step.eval() % self.args.save_freq == 0:
                    print('Saving the model..')
                    checkpoint_path = os.path.join(self.args.model_dir, self.args.model_name)
                    self.model.save(self.sess, checkpoint_path, global_step=self.model.global_step)
                    # json.dump(self.model.args,
                    #           open('%s-%d.json' % (checkpoint_path, self.model.global_step.eval()), 'wb'),
                    #           indent=2)
            # Increase the epoch index of the model
            self.model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(self.model.global_epoch_step.eval()))

        print('Saving the last model..')
        checkpoint_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.model.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        # json.dump(self.model.args,
        #           open('%s-%d.json' % (checkpoint_path, self.model.global_step.eval()), 'wb'),
        #           indent=2)

        print('Training Terminated')