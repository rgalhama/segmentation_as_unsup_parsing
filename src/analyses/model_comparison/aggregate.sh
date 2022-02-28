modelspath=`echo  ~/Research/Projects/backPred/3rdParty/diora/diora/log/`


aggregate () {

  exp=$1
  reps=$2
  mseedinfix=$3
  stseed=$4
  testtype=$5

  fname=${modelspath}/${exp}_broken_reps${reps}_seed${stseed}_??mseed${mseedinfix}??_${testtype}/eval.txt

#  #Grep micro
#  metric="micro"
#  echo -e "wtype\tmseed\tstseed\tprecision\trecall\tf1" > ${exp}_${metric}.csv
#  cat $fname | grep "^0.*" | tr -s " " | sed -e 's/^0 //' | tr ' ' \\t >> ${exp}_${metric}.csv

  #Grep accuracy
  echo "mseed stseed accuracy"> ${exp}_accuracy.txt
  tail -n 1 $fname | sed 's/accuracy\:  //' >> ${exp}_accuracy.txt
  cat ${exp}_accuracy.txt | grep -v "^=" | grep -v "^$" > aux
  mv aux ${exp}_accuracy.csv

}


mseedinfix=1650 #mseed structure:
                # xxyyzzaa: arch+loss combination;
                # yy: hidden layer size;
                # zz: epochs trained;
                # aa: id individual simulation

stseed=11002    #stseed structure
                # 1: PD exps
                # 10: max sentence length
                # 02: attempt (first deviated too much from French)

testtype="test" #test or testontrain
aggregate "fw1" 115 $mseedinfix $stseed $testtype
aggregate "bw1" 115 $mseedinfix $stseed $testtype
aggregate "fw2" 70  $mseedinfix $stseed $testtype
aggregate "bw2" 70  $mseedinfix $stseed $testtype

testtype="testontrain" #test or testontrain
aggregate "fw1" 115 $mseedinfix $stseed $testtype
aggregate "bw1" 115 $mseedinfix $stseed $testtype
aggregate "fw2" 70  $mseedinfix $stseed $testtype
aggregate "bw2" 70  $mseedinfix $stseed $testtype


