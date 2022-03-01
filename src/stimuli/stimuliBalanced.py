# coding:utf-8
"""
Name : sentence_lengths.py.py
Author : Raquel G. Alhama
Desc: Produces stimuli as described in Onnis & Thiessen 2013

Description from paper:
"a template sequence of 711 letter
symbols was generated according to the rules of a stochastic Markovian grammar chain
The process started by choosing one of eight possible symbols (X, Y, A, B, C, D, E, F) at random,
and then generating the next symbol according to the probabilistic sequencing rules specified in Table 3 "

"""

import random, os
import numpy as np
from src.stimuli.sen_len_distr.mean_sentence_lengths import SentenceLengthDistr

class StimuliBalanced:


    def __init__(self, seed):
        random.seed(seed)
        self.symbols = "X,Y,A,B,C,D,E,F".split(',')
        self.states = np.arange(0,len(self.symbols))
        self.w2i=dict(zip(self.symbols,self.states))
        self.i2w=dict(zip(self.states,self.symbols))
        #self.bos, self.eos  = len(self.symbols)+1,len(self.symbols)+2
        self.T = np.array([[0, 0, 0, 0, .33, 0, 0.34, 0.33],
                      [0, 0, .34, 0.36, .30, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0]])

    def generate(self, n, idx=False):
        #Starting state
        act = random.choice(self.states)
        sequence = [act]
        #Generate the rest of states
        for _ in range(n-1):
            nxt = random.choices(self.states, weights=self.T[act, :])
            act=nxt[0]
            sequence.append(act)

        if idx:
            return sequence
        else:
            return [self.i2w[s] for s in sequence]


    def get_hilo_seqs(self, idx=False):
        seqs = [['D', 'Y'],
                ['E', 'Y'],
                ['F', 'Y'],
                ['A', 'X'],
                ['B', 'X'],
                ['C', 'X']]
        if idx:
            seqs = [[self.w2i[w1], self.w2i[w2]] for w1, w2 in seqs]
        return seqs

    def get_lohi_seqs(self, idx=False):
        seqs = [['X', 'D'],
                ['X', 'E'],
                ['X', 'F'],
                ['Y', 'A'],
                ['Y', 'B'],
                ['Y', 'C']]
        if idx:
            seqs = [[self.w2i[w1], self.w2i[w2]] for w1, w2 in seqs]
        return seqs

    def generate_broken_stream(self, length, N, s, maxlength=10):
        stream =self.generate(length, False)
        broken_streams = []
        while len(stream) > 0:
            r = int(random.gauss(N, s))
            while r>maxlength or r<4: #we set a limit for too long sentences
                r = int(random.gauss(N, s))
            broken_streams.append(stream[:r])
            stream = stream[r:]

        return broken_streams

    def create_brokenstream_datasets(self, repetitions, N, s, seed):
        fnames, streams = [], []
        exts = [".train", ".test"]
        for i in range(len(exts)):
            ftemplate = "balanced_broken_reps%i_seed%i" % (repetitions[i], seed)
            fnames.append(ftemplate+exts[i])
            streams.append(self.generate_broken_stream(repetitions[i], N, s))

        for i,f in enumerate(fnames):
            with open(f, "w") as fh:
                for seq in streams[i]:
                    fh.write(" ".join(seq))
                    fh.write("\n")
            print("Output file: %s"%(os.path.join(os.getcwd(),f)))

if __name__ == "__main__":
    seed=211001
    sl=SentenceLengthDistr('fr')
    stim=StimuliBalanced(seed)
    stim.create_brokenstream_datasets((711, 711, 711), sl.mean, sl.sdev, seed)

