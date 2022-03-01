For exact replication of ACL-2022 paper, use these seeds:

For stimuli: stseed=11002

stseed structure (xyyzz):
- x:  id exps from paper (P&D)
- yy: max sentence length
- zz: attempt (first randomization deviated too much from sentence length distribution in French)

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
