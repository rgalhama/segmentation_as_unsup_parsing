# coding:utf-8
"""
Name : learnt_distributions.py
Author : Raquel G. Alhama
Desc:
"""
import sys, os, argparse
import pandas as pd
import json_lines
from src.metrics.parsing_metrics import chunks_from_tree
from analyses.learnt_distributions.graphs import plot_wpw_distr
from src.stimuli.stimuliPD import StimuliPD


def main(exp, treefile, mseed, stseed, outputgraph=False):

    #Gather predicted constituents
    predicted=[]
    with json_lines.open(treefile, "r") as fh:
        for line in fh:
            tree = line["tree"]
            p =chunks_from_tree(tree, [])
            p = [a+b for a,b in p]
            predicted.extend(p)

    #Get test items from the experiment
    stim=StimuliPD(exp, stseed)
    stwords = [(a+b).lower() for a,b in stim.get_words_test()]
    stpartwords = [(a+b).lower() for a,b in stim.get_partwords()]
    testitems=stwords+stpartwords

    #Get counts for test items
    d={s:predicted.count(s) for s in testitems}
    d = dict(sorted(d.items(), key=lambda x:x[1], reverse=True))

    #Convert to dataframe
    recs=[]
    for k,v in d.items():
        recs.append({"mseed":mseed, "stseed": stseed, "exp":exp, "item":k, "isWord":(k in stwords), "counts":v})
    df=pd.DataFrame.from_records(recs)
    df.to_csv("counts_{}_{}_{}.csv".format(exp, mseed, stseed), index=False)


    #Compute correct responses in test
    result = stim.test(d)
    print("accuracy: ", mseed, stseed, result)

    #Plot
    if outputgraph:
        plot_wpw_distr(d, stwords, stpartwords, fname="{}_{}_{}_wpw.png".format(exp, mseed, stseed))





if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    if len(sys.argv) <= 1:
        #Defaults
        print("No arguments provided. Assuming debugging mode with default parameters.")

        for mseed in range(66165001, 66165030+1):
            stseed=11002
            exp="bw1"
            reps=115 #115 for exp 1, 70 for exp 2
            parser.set_defaults(exp=exp,
                                stseed=stseed,
                                mseed=mseed,
                                treefile=os.path.expanduser('~/Research/Projects/backPred/3rdParty/diora/diora/log/'
                                                            '{}_broken_reps{}_seed{}_mseed{}_testontrain/parse.jsonl'.format(exp,reps, stseed, mseed)))
            args = parser.parse_args()
            main(args.exp, args.treefile, args.mseed, args.stseed)
    else:
        parser.add_argument('--exp', type=str, choices=["fw1", "fw2", "bw1", "bw2", "balanced"])
        parser.add_argument('--treefile', help="Path to file with parsed trees")
        parser.add_argument('--mseed', type=int)
        parser.add_argument('--stseed', type=int)

    args = parser.parse_args()
    main(args.exp, args.treefile, args.mseed, args.stseed)
