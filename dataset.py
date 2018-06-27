from __future__ import division

import math
import random

import numpy as np
import constants as Constants

class Dataset(object):
    def __init__(self, data, batchSize, eval=False):
        self.src = data["src"]
        self.tgt = data["tgt"]
        self.pos = data["pos"]
        assert(len(self.src) == len(self.tgt))

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.eval = eval

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [len(x) for x in data]
        max_length = max(lengths)
        # out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        out = np.zeros([len(data), max_length],dtype=np.int32)
        # out = [[0]*max_length]*len(data)
        out.fill(Constants.PAD)
        for i in range(len(data)):
            data_length = len(data[i])
            offset = max_length - data_length if align_right else 0
            out[i][offset:data_length] = data[i]
            # out[i].narrow(0, offset, data_length).copy_(data[i])
        out = out.tolist()
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, src_lengths = self._batchify(self.src[index*self.batchSize:(index+1)*self.batchSize],
            include_lengths=True)

        tgtBatch, tgt_lengths = self._batchify(self.tgt[index*self.batchSize:(index+1)*self.batchSize], include_lengths=True)

        # within batch sort by decreasing length.
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch, tgtBatch, tgt_lengths)
        batch, src_lengths = zip(*sorted(zip(batch, src_lengths), key=lambda x: -x[1]))

        indices, srcBatch, tgtBatch, tgt_lengths = zip(*batch)

        return np.array(srcBatch), np.array(src_lengths), np.array(tgtBatch), np.array(tgt_lengths)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.pos))
        random.shuffle(data)
        self.src, self.tgt, self.pos = zip(*data)

    def restore_pos(self, sents):
        sorted_sents = [None] * len(self.pos)
        for sent, idx in zip(sents, self.pos):
          sorted_sents[idx] = sent
        return sorted_sents