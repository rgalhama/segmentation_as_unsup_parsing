# About

Code for ACL-2022 paper:

Alhama, Raquel G. (2022). Segmentation as Unsupervised Constituency Parsing. 
To appear in *Proceedings of 60th Annual Meeting of the Association for Computational Linguistics*.




## Howto

The stimuli is provided in data/. It can be regenerated with the stimuliPD.py script. 

Run Bash scripts (/src/scripts/*.sh) for training, test and evaluation. 

Scripts in src/analyses can be used to reproduce figures and analyses reported in the ACL2022 paper.

All the reported simulations are based on my forked version of DIORA, which removes dependency on pre-trained embeddings.



For exact replication, use these seeds:

For stimuli: stseed=11002    

stseed structure (xyyzz):
- x: id exps from paper (P&D)
- yy: max sentence length
- zz: attempt (first deviated too much from sentence length distribution in French)

For models: mseed=xx1650aa

mseed structure (xxyyzzaa):
- xx: arch+loss combination (listed below)
- yy: hidden layer size; reported: 16
- zz: epochs trained; reported: 50
- aa: id individual simulation; reported: 01 to 30

xx:
- 33: MLP + Margin
- 44: TreeLSTM + Softmax
- 55: MLP + Softmax
- 66: TreeLSTM + Margin 
- 77: MLP-shared + Softmax
- 88: MLP-shared + Margin

## Whyto

goto #About