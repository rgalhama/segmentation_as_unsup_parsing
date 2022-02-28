#!/usr/bin/env bash

#Test trained models on training data (emulating experiments with humans)

#Set all the paths
source pre_paths.sh


if [ $# -eq 0 ]
  then

    echo "Running experiment with hardcoded params:"

    #Perruchet and Desaulty:
    # bw1, bw2, fw1, fw2
    exp='bw2'
    reps=70 #115 for exp 1
    #Model seed
    seed=66165001
    #stimuli seed:
    stseed=11002


else
  exp=$1
  reps=$2
  seed=$3
  stseed=$4
fi

echo "Simulation $exp $exp_name"
stimuli="${exp}_broken_reps${reps}_seed${stseed}"
model_name="${stimuli}_mseed${seed}"


cd $srcpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# Test on training data (assumes trained model)
python diora/scripts/parse.py \
    --data_type txt \
    --load_model_path ${modelspath}/${model_name}/model_periodic.pt \
    --model_flags ${modelspath}/${model_name}/flags.json \
    --validation_path "${datapath}/${stimuli}.train" \
    --validation_filter_length 0 \
    --emb none \
    --experiment_name "${model_name}_testontrain" \
    --seed ${seed}



#3. Evaluate

#Evaluate with parsing metrics
cd $bwfwsrcpath
echo $bwfwsrcpath
export PYTHONPATH=$(pwd):$PYTHONPATH
resultspath="${modelspath}/${model_name}_testontrain/eval.txt"
python ${evalpath}/parsing_metrics.py --exp ${exp} \
       --treefile "$modelspath/${model_name}_testontrain/parse.jsonl" \
       --stimuliseed $stseed \
       --modelseed $seed \
       > $resultspath


# Evaluate with test metric used in original paper (accuracy)
python ${evalpath}/learnt_distributions.py \
       --exp $exp \
       --treefile "$modelspath/${model_name}_testontrain/parse.jsonl" \
       --stseed $stseed \
       --mseed $seed \
       >> $resultspath

cd ${modelspath}
echo "Results at ${resultspath}"
cat $resultspath

conda deactivate
