# coding:utf-8
"""
Name :
Author : Raquel G. Alhama
Desc:  Metrics to evaluate segmentation as parsing
"""
import sys, os
import re
import json_lines
import pandas as pd
import argparse
from src.stimuli.stimuliPD import StimuliPD
#from src.stimuli.stimuliBalanced import StimuliBalanced #Onnis&Thiessen stimuli


def chunks_from_tree(tree, chunks=[]):
    """
    Depth-first search to find chunks (lower-level constituents) in the tree.
    :param tree:
    :param chunks:
    :return:
    """
    if len(tree) == 0:
        return chunks
    if len(tree) == 1:
        return []
    left, right = tree[0], tree[1]
    if type(left) != list and type(right) != list:
         return [(left, right)]
    left = chunks_from_tree(tree[0], chunks)
    right = chunks_from_tree(tree[1], chunks)
    return left+right


def overlap_chunks(gold, predicted):
    gs = set([a+b for a,b in gold])
    ps = set([a+b for a,b in predicted])
    correct = ps.intersection(gs)
    return correct

def tree2seq(tree, seq=""):
    #Tree to inorder sequence
    if type(tree) == str:
        return tree #it's a leaf
    else:
        return tree2seq(tree[0]) + tree2seq(tree[1])


def seq2phrases(seq):
    g = re.findall(r"..", seq)
    h = re.findall(r"..",seq[1:])
    return g+h


def get_gold(vocab_phrases, tree):

    gs = set([a+b for a,b in vocab_phrases])
    tseq =tree2seq(tree)
    tph = seq2phrases(tseq)
    goldintree = gs.intersection(set(tph))
    return goldintree

def get_counts(vocab_phrases, tree):

    gold = get_gold(vocab_phrases, tree)
    predicted = chunks_from_tree(tree, [])
    correct=overlap_chunks(vocab_phrases, predicted)

    return len(gold), len(predicted), len(correct)

def eval(n_gold, n_predicted, n_correct):
    try:
        precision = n_correct / n_predicted
    except ZeroDivisionError:
        precision = 0
    try:
        recall = n_correct / n_gold
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1


def report(args, wr, pwr):

    #Report
    print("Experiment: %s"%args.exp)
    print("Path: %s"%args.treefile)
    if args.exp == 'balanced':
        print("HiLo:")
    else:
        print("Words:")
    print(wr)
    if args.exp == 'balanced':
        print("LoHi:")
    else:
        print("Partwords:")
    print(pwr)

def macro(treefile, words, partwords):
    """ Precision, Recall and F-1 computed from average over individual p, r, f1 (per sentence)"""
    wr, pwr = [], []
    with json_lines.open(treefile, "r") as fh:
        for line in fh:
            tree = line["tree"]
            p,r,f1=eval(*get_counts(words, tree))
            wr.append({"precision": p, "recall": r, "f1": f1})
            p,r,f1=eval(*get_counts(partwords, tree))
            pwr.append({"precision": p, "recall": r, "f1": f1})

    wdf=pd.DataFrame.from_dict(wr, orient="columns")
    pwdf=pd.DataFrame.from_dict(pwr, orient="columns")
    return wdf.mean(), pwdf.mean()

def micro(treefile, words, partwords):
    """ Precision, Recall and F-1 are computed over summed predictions."""
    wr, pwr = [], []
    totals=dict(zip(('wgold', 'wpredicted', 'wcorrect', 'pwgold', 'pwpredicted', 'pwcorrect'),[0]*6))
    with json_lines.open(treefile, "r") as fh:
        for line in fh:
            tree = line["tree"]
            w_gold, w_predicted, w_correct = get_counts(words, tree)
            pw_gold, pw_predicted, pw_correct = get_counts(partwords, tree)
            totals['wgold']+=w_gold
            totals['wpredicted']+=w_predicted
            totals['wcorrect']+=w_correct
            totals['pwgold']+=pw_gold
            totals['pwpredicted']+=pw_predicted
            totals['pwcorrect']+=pw_correct
    p,r,f1=eval(totals['wgold'], totals['wpredicted'], totals['wcorrect'])
    wr.append({"index":"words", "mseed": args.modelseed, "stseed":args.stimuliseed, "precision": p, "recall": r, "f1": f1})
    p,r,f1=eval(totals['pwgold'], totals['pwpredicted'], totals['pwcorrect'])
    pwr.append({"index":"partwords", "mseed": args.modelseed, "stseed":args.stimuliseed,"precision": p, "recall": r, "f1": f1})

    wdf=pd.DataFrame.from_dict(wr)
    pwdf=pd.DataFrame.from_dict(pwr)
    return wdf, pwdf


def get_items_stimuli(exp):

    #Get words/partwords from the stimuli
    if exp == "balanced":
        # For future work using Onnis&Thiessen's stimuli:
        # st = StimuliBalanced(1)
        # hilo = [[a.lower(), b.lower()] for [a,b] in st.get_hilo_seqs()]
        # lohi = [[a.lower(), b.lower()] for [a,b] in st.get_lohi_seqs()]
        # words=hilo
        # partwords=lohi
        raise NotImplemented
    else:
        st = StimuliPD(exp, 1)
        words = [[a.lower(), b.lower()] for [a,b] in st.get_words_test()]
        partwords= [[a.lower(), b.lower()] for [a,b] in st.get_partwords()]

    return words, partwords

def main(args):

    #Get correct and incorrect constituents (words/partwords)
    words, partwords = get_items_stimuli(args.exp)

    #Macro: precision, recall and f1 per sentence, then average
    wr, pwr = macro(args.treefile, words, partwords)
    print("MACRO:")
    report(args, wr, pwr)
    #Micro:
    wr, pwr = micro(args.treefile, words, partwords)
    print("\nMICRO:")
    report(args, wr, pwr)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    if len(sys.argv) <= 1:
        #Defaults
        print("No arguments provided. Assuming debugging mode with default parameters.")
        parser.set_defaults(exp="bw2",modelseed=66165001, stimuliseed=11002,
                            treefile=os.path.expanduser('~/Research/Projects/'
                            'backPred/3rdParty/diora/diora/log/bw2_broken_reps70_seed11002_mseed66165001_test/parse.jsonl'))
    else:
        parser.add_argument('--exp', type=str, choices=["fw1", "fw2", "bw1", "bw2", "balanced"])
        parser.add_argument('--treefile', help="Path to file with parsed trees")
        parser.add_argument('--stimuliseed', type=int)
        parser.add_argument('--modelseed', type=int)
    args = parser.parse_args()


    main(args)

