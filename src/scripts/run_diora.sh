#!/usr/bin/env bash


#Set all the paths
source pre_paths.sh


if [ $# -eq 0 ]
  then

    echo "Running experiment with hardcoded params:"

    #Perruchet and Desaulty:
    # bw1, bw2, fw1, fw2; adjust reps
    exp='bw2'
    reps=70
    #model seed:
    seed=66105001
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

batchsize=20

# 1. Train DIORA on fw/bw languages from Perruchet&Desaulty
# Note: validation path is required but it's not really used for training
python diora/scripts/train.py \
    --batch_size ${batchsize} \
    --hidden_dim 16 \
    --arch 'mlp' \
    --reconstruct_mode 'margin' \
    --max_epoch 50 \
    --data_type txt \
    --log_every_batch 1 \
    --train_filter_length 0 \
    --validation_filter_length 0 \
    --train_path $datapath/${stimuli}.train \
    --emb none \
    --experiment_name ${model_name} \
    --save_latest 1 \
    --save_distinct 2000 \
    --save_after 1 \
    --seed ${seed} \
    --cuda \
    --normalize 'unit' \
    --validation_path $datapath/${stimuli}.train \
#    --multigpu

# Notes on saving models:
# save_after: Checkpoints will only be saved after N gradient updates have been applied.
# save_latest: Every N gradient updates, a checkpoint will be saved called `model_periodic.pt`.
# This is the final model.
# save_distinct: Every N gradient updates, a checkpoint will be saved called `model.step_${N}.pt`.
# This is meant for intermediate models.

# 2. Test
python diora/scripts/parse.py \
    --batch_size $batchsize \
    --data_type txt \
    --load_model_path ${modelspath}/${model_name}/model_periodic.pt \
    --model_flags ${modelspath}/${model_name}/flags.json \
    --validation_filter_length 0 \
    --emb none \
    --experiment_name ${model_name}_test \
    --seed ${seed} \
    --validation_path "${datapath}/${stimuli}.test" \


#3. Evaluate

cd $bwfwsrcpath
echo $bwfwsrcpath
export PYTHONPATH=$(pwd):$PYTHONPATH
resultspath="${modelspath}/${model_name}_test/eval.txt"
python ${evalpath}/parsing_metrics.py --exp ${exp} \
       --treefile $modelspath/${model_name}_test/parse.jsonl \
       --stimuliseed $stseed \
       --modelseed $seed \
       > ${resultspath}

# Evaluate with test metric used in original paper (accuracy)
python ${evalpath}/learnt_distributions.py \
       --exp $exp \
       --treefile "$modelspath/${model_name}_test/parse.jsonl" \
       --stseed $stseed \
       --mseed $seed \
       >> ${resultspath}

cd ${modelspath}
evalpath="${modelspath}/${model_name}_test/eval.txt"
echo "Results at ${evalpath}"
cat $evalpath

conda deactivate
