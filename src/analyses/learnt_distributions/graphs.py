# coding:utf-8
"""
Name : graphs.py.py
Author : Raquel G. Alhama
Desc:
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_wpw_distr(d, stwords, stpartwords, fname="distr_wpw.png"):

    colors, labels = [], []
    items=list(d.keys())
    for k in items:
        if k in stwords:
            colors.append('green')
            labels.append('word')
        elif k in stpartwords:
            colors.append('red')
            labels.append('partword')
        else: #not part of test stimuli
            del d[k]

    df = pd.DataFrame({"x":d.keys(),
                       "y":d.values(),
                       "color":colors,
                       "label":labels})
    for i, dff in df.groupby("color"):
        label = 'word' if dff['color'].iloc[0] == 'green' else 'partword'
        plt.scatter(dff['x'], dff['y'], s=50, c=dff['color'],
                    edgecolors='none', label=label)

    plt.legend()
    plt.xlabel("Test Stimuli")
    plt.ylabel("Times identified as constituent")
    plt.savefig(fname)
    plt.clf()
