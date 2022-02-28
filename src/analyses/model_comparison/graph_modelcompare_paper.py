# coding:utf-8
"""
Name : graph_modelcompare_paper.py
Author : Raquel G. Alhama
Desc:
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def namer(s):
    if s == '33':
        return 'MLP-margin'
    if s == '44':
        return 'TreeLSTM-softmax'
    if s == '55':
        return 'MLP-softmax'
    if s == '66':
        return 'TreeLSTM-margin'
    if s == '77':
        return 'MLPshared-softmax'
    if s == '88':
        return 'MLPshared-margin'

def expfull(exp):
    nmbr=exp[-1]
    dir=exp[:2]
    return "Experiment {}:{}".format(nmbr,dir)

def expname(exp):
    nmbr=exp[-1]
    return "Experiment {}".format(nmbr)

def direction(exp):
    return "Forward" if exp[0] == "f" else "Backward"


#--------------------

expsdf=dict(zip(['fw1', 'bw1', 'fw2', 'bw2'], [None]*4))

for exp in expsdf.keys():
    expsdf[exp]=pd.read_csv("accuracy_xx1650xx_{}.csv".format(exp), sep = " ", index_col=None)

fig, axes = plt.subplots(2,2, sharey=True, figsize=(14,14))
for i,(exp,df) in enumerate(expsdf.items()):
    df['mseed']=df['mseed'].astype(str)
    pat=r"(..)(.*)"
    df[['model', '-mseed']]=df.mseed.str.extract(pat, expand=True)
    df["model"]=df['model'].map(namer)
    #df=df.sort_values('model')
    ax_i=int(exp[-1])-1
    ax_j=0 if exp[0]=='f' else 1
    sns.boxplot(x="model", y="accuracy",
                       data=df, palette="Set3", dodge=True, ax=axes[ax_i, ax_j])
    axes[ax_i,ax_j].set_xticklabels(axes[ax_i,ax_j].get_xticklabels(),rotation = 20)
    axes[ax_i,ax_j].axhline(y=0.5, color='black', ls="--")
    axes[ax_i,ax_j].set_xlabel('')
    axes[ax_i,0].set_ylabel(expname(exp), fontsize=16)
    axes[0,ax_j].set_title(direction(exp), fontsize=16)
    fig.tight_layout()

plt.savefig('compare_models_pd.png')
