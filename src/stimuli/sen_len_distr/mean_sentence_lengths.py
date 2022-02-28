# coding:utf-8
"""
Name : sentence_lengths.py.py
Author : Raquel G. Alhama
Desc: Check distribution of sentence length
"""


import numpy as np
#import matplotlib.pyplot as plt


class SentenceLengthDistr:
    """ Container for previously computed empirical mean and std of sentence lengths.
        Computed on data from OpenSubtitles (replicate with function below; this is just a
        convenient hand-coded class).
    """


    def __init__(self, language):
        if language == 'fr':
            self.mean=5.93
            self.sdev=4.56
        elif language == 'en':
            self.mean=5.79
            self.sdev=4.65
        elif language == 'ja':
            self.mean=6.85
            self.sdev=4.45



def get_len_distr(language):
    '''Compute the distribution of sentence length (use after get_sentence_length.sh)'''
    fname="%s.sentence.len.txt"%language
    ls=[]
    with open(fname, "r") as fh:
        for line in fh:
            l=line.strip()
            if l:
                ls.append(int(l))

    #plt.hist(ls, bins=10)
    #plt.show()
    return np.mean(ls), np.std(ls)

if __name__=="__main__":
    N,s=get_len_distr("fr")
    print(N,s)

