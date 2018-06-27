from __future__ import division
import math
import time
import random
import numpy as np
import sys

class Evaluator():
    def __init__(self, sess, model, test_data, dicts, args):
        self.sess = sess
        self.model = model
        self.test_data = test_data
        # self.evaluator = lib.Evaluator(model, metrics, dicts, opt)
        self.dicts = dicts
        self.args = args

    def eval(self):
        print('Evaluating..')
        try:
            if self.args.write_n_best:
                fout = [open("%s_%d" % (self.args.decode_output, k), 'w') for k in range(self.args.beam_width)]
            else:
                fout = [open(self.args.decode_output, 'w')]
            for i in range(len(self.test_data)):
                batch = self.test_data[i]
                source, source_len, target, target_len = batch
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = self.model.predict(self.sess, encoder_inputs=source, encoder_inputs_length=source_len)
                # Write decoding results
                for k, f in reversed(list(enumerate(fout))):
                    for seq in predicted_ids:
                        # f.write(str(data_utils.seq2words(seq[:, k], self.dicts['tgt'])) + '\n')
                        sent = [self.dicts["tgt"].getLabel(w) for w in seq[:, k]]
                        f.write(' '.join(sent) + '\n')
                    if not self.args.write_n_best:
                        break
                # print('  {}th line decoded'.format(idx * FLAGS.decode_batch_size))

            print('Decoding terminated')
        except IOError:
            pass
        finally:
            [f.close() for f in fout]