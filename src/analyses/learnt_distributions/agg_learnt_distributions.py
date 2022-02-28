# coding:utf-8
"""
Name : agg_learnt_distributions.py.py
Author : Raquel G. Alhama
Desc: Creates Figure 3 acl2002-paper
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import src.analyses.learnt_distributions as ld

def counts_by_model(stseed, exp, reps, mrange, outputgraph=False):
    for mseed in mrange:
        treefile=os.path.expanduser('~/Research/Projects/backPred/3rdParty/diora/diora/log/'
                           '{}_broken_reps{}_seed{}_mseed{}_testontrain/parse.jsonl'.format(exp,reps, stseed, mseed))

        ld.main(exp, treefile, mseed, stseed, outputgraph=False)

if __name__=="__main__":
    stseed=11002
    exp="bw2"
    reps=70 #115 for exp 1, 70 for exp 2
    mrange=range(66165001, 66165030+1)
    m_ind='661650xx'

    # Create individual files with counts
    counts_by_model(stseed, exp, reps, mrange)

    #Aggregate in one file
    # head -n1 counts_bw1_66165001_11002.csv > all.csv
    # cat counts*.csv | grep -v ^mseed >> all.csv


    # Load, aggregate, create graph
    df=pd.read_csv("{}_{}/all.csv".format(stseed, m_ind))

    pat=r"(..)(.*)"
    df[['direction', 'Experiment']]=df.exp.str.extract(pat, expand=True)
    df=df.sort_values(['isWord', 'Experiment'])

    #t-test
    def f(x):
        df_w=x.query("isWord == True")
        df_pw=x.query("isWord == False")
        t, p = ttest_ind(df_w['counts'], df_pw['counts'])
        return pd.Series({'t':t, 'p':p})
    gdf=df.groupby("exp").apply(f)
    gdf.to_csv("t_test.csv", float_format='%.2f')
    exit()


    #Graph
    fig, axes = plt.subplots(2,2, sharey=False, sharex=False, figsize=(14,14))
    g=sns.FacetGrid(df, col="direction", row="Experiment", sharex=False)
    g.map_dataframe(sns.barplot, x="item", y="counts", hue="isWord", palette="husl")
    # axes[ax_i,ax_j].set_xticklabels(axes[ax_i,ax_j].get_xticklabels(),rotation = 20)
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    g.add_legend(bbox_to_anchor= (.98, .4),)
    for t, l in zip(g._legend.texts, ['partwords', 'words']):
        t.set_text(l)
    plt.tight_layout()
    plt.savefig("test_item_distr_{}_{}.png".format(stseed, m_ind))


