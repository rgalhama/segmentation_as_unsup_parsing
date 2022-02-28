# coding:utf-8
"""
Name :
Author : Raquel G. Alhama
Desc:  Produces stimuli described in Perruchet&Desaulty
"""
import os, sys, inspect

import pandas as pd

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams.update({'font.size': 15})
from itertools import chain
import re
from src.stimuli.sen_len_distr.mean_sentence_lengths import SentenceLengthDistr
from analyses.learnt_distributions.graphs import plot_wpw_distr

def seq2phrases(seq):
    g = re.findall(r"..", seq)
    h = re.findall(r"..", seq[1:])
    return g + h


class StimuliPD:
    def __init__(self, exp, seed):
        random.seed(seed)
        exps = ["fw1", "bw1", "fw2", "bw2"]
        assert (exp in exps)
        self.exp = exp
        self.seed=seed
        self.symbols = "A,B,C,D,E,F,G,H,I,X,Y,Z".split(',')
        self.class_A = "A,B,C,D,E,F,G,H,I".split(',')
        self.class_X = "X,Y,Z".split(',')
        self.states = np.arange(0, len(self.symbols))
        self.w2i = dict(zip(self.symbols, self.states))
        self.i2w = dict(zip(self.states, self.symbols))
        self.delayIdx = len(self.symbols)
        # self.bos, self.eos  = len(self.symbols)+1,len(self.symbols)+2

        if exp.startswith("fw"):
            self.initial_symbols = self.class_A
            self.T = self.fwdT()
        else:  # bw
            self.initial_symbols = self.class_X
            self.T = self.bwdT()

    def fwdT(self):
        T = np.zeros((len(self.symbols), len(self.symbols)))
        T[self.w2i['A'], self.w2i['X']] = 1
        T[self.w2i['B'], self.w2i['X']] = 1
        T[self.w2i['C'], self.w2i['X']] = 1
        T[self.w2i['D'], self.w2i['Y']] = 1
        T[self.w2i['E'], self.w2i['Y']] = 1
        T[self.w2i['F'], self.w2i['Y']] = 1
        T[self.w2i['G'], self.w2i['Z']] = 1
        T[self.w2i['H'], self.w2i['Z']] = 1
        T[self.w2i['I'], self.w2i['Z']] = 1
        T[self.w2i['X'], :len(self.class_A)] = [1 / len(self.class_A)] * len(self.class_A)
        T[self.w2i['Y'], :len(self.class_A)] = [1 / len(self.class_A)] * len(self.class_A)
        T[self.w2i['Z'], :len(self.class_A)] = [1 / len(self.class_A)] * len(self.class_A)
        return T

    def bwdT(self, ):
        T = np.zeros((len(self.symbols), len(self.symbols)))
        T[self.w2i['X'], self.w2i['A']] = 1 / 3
        T[self.w2i['X'], self.w2i['B']] = 1 / 3
        T[self.w2i['X'], self.w2i['C']] = 1 / 3
        T[self.w2i['Y'], self.w2i['D']] = 1 / 3
        T[self.w2i['Y'], self.w2i['E']] = 1 / 3
        T[self.w2i['Y'], self.w2i['F']] = 1 / 3
        T[self.w2i['Z'], self.w2i['G']] = 1 / 3
        T[self.w2i['Z'], self.w2i['H']] = 1 / 3
        T[self.w2i['Z'], self.w2i['I']] = 1 / 3
        row = np.zeros(len(self.symbols))
        row[self.w2i['X']] = 1 / len(self.class_A)
        row[self.w2i['Y']] = 1 / len(self.class_A)
        row[self.w2i['Z']] = 1 / len(self.class_A)
        for s in self.class_A:
            T[self.w2i[s], :] = row
        return T

    def generate(self, n, idx):
        """
        Generate stimuli according to reported experiments (with variable n).
        n is the number of repetitions of each word (in exp 1) or of most words (in exp 2, where 3 words appear 3 times
        more than the rest)
        Original value for n: 115 (exp 1) and 70 (exp 2)
        :param n: length of the stream (in words, i.e. syllable duplets)
        :param idx:
        :return:
        """
        stream = []
        if self.exp[-1] == '1':  #
            for _ in range(n):
                stream.extend(self.get_words())
        elif self.exp[-1] == '2':
            # In experiment 2, some words appear 3 times
            if self.exp.startswith("fw"):
                extra_seqs = [['A', 'X'], ['D', 'Y'], ['G', 'Z']]
            else:
                extra_seqs = [['X', 'A'], ['Y', 'D'], ['Z', 'G']]
            for _ in range(n):
                stream.extend(self.get_words())
                for _ in range(2): #add two more repetitions of each extra seq
                    stream.extend(extra_seqs)

        # Shuffle
        random.shuffle(stream)
        # Substitute letters with indexs if required
        if idx:
            stream = self.stream_to_index(stream)
        # Flatten
        stream = list(chain.from_iterable(stream))

        return (stream)

    def generate_markov_chain(self, n):
        """
        Alternative method: produce a stream based purely on the FSA for this language.
        Contrary to the streams used in the published experiments, this function
        produces the stream without any control for word/partword frequency.

        :param n: length of the stream (in symbols)
        :return:
        """
        # Starting state
        act = random.choice([self.w2i[s] for s in self.initial_symbols])
        sequence = [act]
        # Generate the rest of states
        for _ in range(n - 1):
            nxt = random.choices(self.states, weights=self.T[act, :])
            act = nxt[0]
            sequence.append(act)

        return sequence

    def get_words(self, idx=False):
        if self.exp.startswith("fw"):
            seqs = [['A', 'X'],
                    ['B', 'X'],
                    ['C', 'X'],
                    ['D', 'Y'],
                    ['E', 'Y'],
                    ['F', 'Y'],
                    ['G', 'Z'],
                    ['H', 'Z'],
                    ['I', 'Z'],
                    ]
        else:  # backward
            seqs = [['X', 'A'],
                    ['X', 'B'],
                    ['X', 'C'],
                    ['Y', 'D'],
                    ['Y', 'E'],
                    ['Y', 'F'],
                    ['Z', 'G'],
                    ['Z', 'H'],
                    ['Z', 'I'],
                    ]
        if idx:
            seqs = [[self.w2i[w1], self.w2i[w2]] for w1, w2 in seqs]
        return seqs

    def get_words_test(self, idx=False, pretty=False):
        """
            Returns the words used in test of original experiment
            (in some cases, that's a subset from the words in the stream!)
        :param idx:
        :return:
        """
        if self.exp == "fw1" or self.exp == "bw1":
            seqs = self.get_words(idx=idx)
        else:  # exp 2 uses a subset of words
            if self.exp.startswith("fw"):
                seqs = [
                    ['B', 'X'],
                    ['C', 'X'],
                    ['E', 'Y'],
                    ['F', 'Y'],
                    ['H', 'Z'],
                    ['I', 'Z'],
                ]
            else:  # backward
                seqs = [
                    ['X', 'B'],
                    ['X', 'C'],
                    ['Y', 'E'],
                    ['Y', 'F'],
                    ['Z', 'H'],
                    ['Z', 'I'],
                ]
        if idx:
            seqs = [[self.w2i[w1], self.w2i[w2]] for w1, w2 in seqs]
        elif pretty:
            seqs = [(a+b).lower() for [a,b] in seqs]
        return seqs

    def get_partwords(self, idx=False, pretty=False):
        """ These are the partwords used in the original 2AFC task (i.e. in the test)
            There exist more partword combinations (i.e. theoretically possible when randomly combining words)
        """
        if self.exp == "fw1":
            seqs = [['X', 'D'],
                    ['X', 'E'],
                    ['X', 'F'],
                    ['Y', 'G'],
                    ['Y', 'H'],
                    ['Y', 'I'],
                    ['Z', 'A'],
                    ['Z', 'B'],
                    ['Z', 'C'],
                    ]
        elif self.exp == "bw1":
            seqs = [['A', 'Y'],
                    ['B', 'Y'],
                    ['C', 'Y'],
                    ['D', 'Z'],
                    ['E', 'Z'],
                    ['F', 'Z'],
                    ['G', 'X'],
                    ['H', 'X'],
                    ['I', 'X'],
                    ]
        elif self.exp == "fw2":  # XA, XD, YD, YG, ZA, and ZG
            seqs = [['X', 'A'],
                    ['X', 'D'],
                    ['Y', 'D'],
                    ['Y', 'G'],
                    ['Z', 'A'],
                    ['Z', 'G'],
                    ]
        elif self.exp == "bw2":  # AX, AY, DY, DZ, GX, and GZ
            seqs = [['A', 'X'],
                    ['A', 'Y'],
                    ['D', 'Y'],
                    ['D', 'Z'],
                    ['G', 'X'],
                    ['G', 'Z'],
                    ]
        if idx:
            seqs = [[self.w2i[w1], self.w2i[w2]] for w1, w2 in seqs]
        elif pretty:
            seqs = [(a+b).lower() for [a,b] in seqs]
        return seqs

    # HiLo/LoHi interface (words map to HiLo or LoHi depending on direction)
    def get_hilo_seqs(self, idx=False, test=False, ):
        if self.exp.startswith("fw"):
            if test:
                return self.get_words_test(idx)
            return self.get_words(idx)
        else:
            return self.get_partwords(idx)

    def get_lohi_seqs(self, idx=False, test=False):
        if self.exp.startswith("fw"):
            return self.get_partwords(idx)
        else:
            if test:
                return self.get_words_test(idx)
            return self.get_words(idx)

    def stream_to_index(self, stream):
        return [[self.w2i[s1], self.w2i[s2]] for s1, s2 in stream]

    def index_to_stream(self, idxstream):
        return [[self.i2w[s1], self.w2i[s2]] for s1, s2 in idxstream]

    def create_datasets(self, nseqs, repetitions):
        fnames = []
        for i in range(3):
            ftemplate = "pd_%s_nseqs%i_reps%i" % (self.exp, nseqs[i], repetitions[i])
            fnames.extend([ftemplate + ext for ext in [".train", ".dev", ".test"]])
        for _, f in enumerate(fnames):
            with open(f, "w") as fh:
                for _ in range(nseqs[i]):
                    stream = self.generate(repetitions[i], False)
                    fh.write(" ".join(stream))
                    fh.write("\n")
            print(" File: %s" % (os.path.join(os.getcwd(), f)))

    def generate_broken_stream(self, length, N, s, maxlength):
        stream = self.generate(length, False)
        broken_streams = []
        while len(stream) > 0:
            r = int(random.gauss(N, s))
            while r > maxlength or r < 4:  # we set a limit for too long sentences
                r = int(random.gauss(N, s))
            broken_streams.append(stream[:r])
            stream = stream[r:]

        return broken_streams

    def write_files(self, fnames, streams):

        for i, f in enumerate(fnames):
            with open(f, "w") as fh:
                for seq in streams[i]:
                    fh.write(" ".join(seq))
                    fh.write("\n")
            print(" File: %s" % (os.path.join(os.getcwd(), f)))

    def report_stats(self, stream):
        seqs = []
        # Gather all bysillabic items
        for line in stream:
            seqs.extend(seq2phrases(''.join(line)))
        # Count
        d = {s: seqs.count(s) for s in seqs}
        # Plot
        words = [a + b for [a, b] in self.get_words_test()]
        fw= [v for k,v in d.items() if k in words]
        partwords = [a + b for [a, b] in self.get_partwords()]
        fpw= [v for k,v in d.items() if k in partwords]
        plot_wpw_distr(d, words, partwords, fname="{}_{}_wpw.png".format(self.exp, self.seed))

        print("\n{} Mean words: {:.2f}({:.2f}) Mean partwords: {:.2f}({:.2f})".
              format(self.exp, np.mean(fw), np.std(fw), np.mean(fpw), np.std(fpw)))

    def create_brokenstream_datasets(self, repetitions, N, s, seed, maxlength, outputfiles=True):
        fnames, streams = [], []
        exts = [".train", ".test"]
        for ext in exts:
            ftemplate = "%s_broken_reps%i_seed%i" % (self.exp, repetitions, seed)
            fnames.append(ftemplate + ext)
            streams.append(self.generate_broken_stream(repetitions, N, s, maxlength))

        self.report_stats(streams[0])

        if outputfiles:
            self.write_files(fnames, streams)

        return streams

    def get_test_stimuli(self):
        pw=self.get_partwords(pretty=True)
        w=self.get_words_test(pretty=True)

        pairs= []

        #Experiment 1 pairs the 9 words with the 9 partwords, 2 times
        #the specific pairing is not described in the paper
        #the only constraint is that the pairs in the two sublists are distinct
        if '1' in self.exp:
            pairing1=list(zip(w,pw))
            pw2=pw[:]
            compare=True
            while compare:
                random.shuffle(pw2)
                compareL=[pw[i]==pw2[i] for i in range(len(pw))]
                compare=True in compareL
            pairing2=list(zip(w,pw2))
            return pairing1+pairing2

        #Experiment 2 is an exhaustive combination of words and partwords (6x6)
        if '2' in self.exp:
            for a in w:
                for b in pw:
                    pairs.append((''.join(a),''.join(b)))

        return pairs

    def test(self, constituents):
        #Create test pairs
        pairs=self.get_test_stimuli()

        #Check against constituents
        correct=[]
        for (w,pw) in pairs:
            if constituents[w.lower()] > constituents[pw.lower()]:
                correct.append(1)
            elif constituents[w.lower()] < constituents[pw.lower()]:
                correct.append(0)
            else:
                correct.append(random.choice([0,1]))

        return np.mean(correct)

    def graph_sentence_length(self, mu, sig, streams, sl_lengths_file=''):

        def poisson(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        x_fr_th = np.linspace(0, 25, 100)
        #plt.plot(x_values, poisson(x_values, mu, sig), label="French (estimated)")
        if sl_lengths_file is not None and sl_lengths_file != '':
            # x_fr = np.fromfile(sl_lengths_file)
            frhist = pd.read_csv(sl_lengths_file, sep=" ", header=None, names=['y', 'x'])



        sls, labels = [], []
        r=[]
        key_to_label={"fw1":"Exp 1: forward", "bw1":"Exp 1: backward","fw2":"Exp 2: forward","bw2":"Exp 2: backward",}
        for key,stream in streams.items():
            sl=[len(s) for s in stream]
            sls.append(sl)
            labels.append(key_to_label[key])
            for s in stream:
                r.append({'length':len(s), 'exp':key})
        sls=np.array(sls)
        df=pd.DataFrame.from_records(r)
        p, ax = plt.subplots()
        #sns.histplot(data=df,  x="length", hue="exp",  multiple='dodge', stat='probability', bins=20)#,
                     # bins=40, common_norm=False, common_bins=False )
        d=plt.hist(sls, bins=9, density=False, histtype='bar', stacked=False, alpha=0.65, label=labels, align='mid', zorder=3)
        ax.locator_params(axis='x', integer=True)
        #Add plot french
        sns.lineplot(x_fr_th, poisson(x_fr_th, mu, sig)*d[0].max(), label="French (estimated)", zorder=2)
        if sl_lengths_file:
            #sns.lineplot(frhist.x, frhist.y, label='French')
            frhist['rescaled']=frhist.y*d[0].max()/frhist.y.max()
            sns.barplot(frhist.x,frhist.rescaled, label='French (empirical)', zorder=1, color='grey', alpha=0.3)

        plt.xlim(0,20)
        plt.xlabel("Sentence length")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()

        plt.savefig("stimuli_{}_slength.png".format(self.seed))


if __name__ == "__main__":
    maxlength=10
    seed=int('1'+str(maxlength)+'02')
    streams={}
    for exp in ("fw1", "bw1", "fw2", "bw2"):
        stim = StimuliPD(exp, seed)
        sl = SentenceLengthDistr('fr')
        #Repetitions depend on experiment (115 for 1,70 for 2)
        reps = 115 if exp[-1] == '1' else 70
        trainstream, teststream = stim.create_brokenstream_datasets(
            reps, sl.mean, sl.sdev, seed, maxlength, outputfiles=False)
        streams[exp]=trainstream

    stim.graph_sentence_length(sl.mean, sl.sdev, streams,
                               #optional:
                               # os.path.expanduser('~/Data_Research/open_subtitles/fr.sentence.length.hist.txt'))
                                '')