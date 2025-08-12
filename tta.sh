#!/bin/bash

SRC_PREFIX="reproduce_src"
LOG_PREFIX="eval_results"

BASE_DATASETS=("cifar10") # "pacs" "tiny-imagenet" "cifar10" "cifar100"
METHODS=("BiTTA")

### TTA ############################
# "TENT" "EATA" "SAR" "SoTTA" "RoTTA" "CoTTA"

### Active TTA baselines ###########
# "SimATTA_BIN" "SimATTA"

### Ours ###########################
# "BiTTA"

SEEDS=(0)
DISTS=(1)
VALIDATIONS=(
            # (1) For running BiTTA and SimATTA (full-label)

            "--log_name log/"  


            # (2) Set --enable_bitta to run TTA baselines in TTA with binary feedback.
            # Exception: For SimATTA, we have SimATTA_BIN method for TTA with binary feedback.
            # Exception: For BiTTA, you can just run BiTTA.
            
            # "--log_name log/ --enable_bitta"


            # (3) Enable --random_setting to test random data stream for PACS.

            # "--log_name log/ --random_setting"
           )


MAIN_SCRIPT="main.py"


### continual adaptation ###########################
TGTS="cont"


echo BASE_DATASETS: "${BASE_DATASETS[@]}"
echo METHODS: "${METHODS[@]}"
echo SEEDS: "${SEEDS[@]}"
GPUS=(0 1 2 3 4 5 6 7) #available gpus
NUM_GPUS=${#GPUS[@]}

sleep 1 # prevent mistake
mkdir raw_logs # save console outputs here


#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=8  #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}


test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & Ours; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for DATASET in "${BASE_DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for validation in "${VALIDATIONS[@]}"; do
        update_every_x="64"
        memory_size="64"
        SEED="0"
        lr="0.001" #other baselines
        weight_decay="0"

        for SEED in "${SEEDS[@]}"; do #multiple seeds
          if [ "${DATASET}" = "pacs" ] || [ "${DATASET}" = "tiny-imagenet" ]; then
            MODEL="resnet18_pretrained"
            CP="--load_checkpoint_path pretrained_weights/${DATASET}/bitta_cp/cp_last.pth.tar"
          elif [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar100" ]; then
            MODEL="resnet18"
            CP="--load_checkpoint_path pretrained_weights/${DATASET}/bitta_cp/cp_last.pth.tar"
          elif [ "${DATASET}" = "imagenet" ]; then
            MODEL="resnet18_pretrained"
            CP=""
          fi

          if [ "${METHOD}" = "Src" ]; then
            EPOCH=0
            #### Train with BN
            for TGT in $TGTS; do
              python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --update_every_x ${update_every_x} --seed $SEED \
                --log_prefix ${LOG_PREFIX}_${SEED} \
                ${validation}  \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          elif [ "${METHOD}" = "SoTTA" ]; then

            lr="0.001"
            EPOCH=1
            loss_scaler=0
            bn_momentum=0.2

            if [ "${DATASET}" = "pacs" ]; then
              high_threshold=0.99
            elif [ "${DATASET}" = "tiny-imagenet" ]; then
              high_threshold=0.33
            fi
            #### Train with BN

            for dist in "${DISTS[@]}"; do
              for memory_type in "HUS"; do
                for TGT in $TGTS; do
                  python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method SoTTA --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                    --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                    --loss_scaler ${loss_scaler} --sam \
                    ${validation} \
                    --high_threshold ${high_threshold} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            done
          elif [ "${METHOD}" = "RoTTA" ]; then
            EPOCH=1
            loss_scaler=0
            lr="0.001"
            bn_momentum=0.05
            #### Train with BN

            for dist in "${DISTS[@]}"; do

              for memory_type in "CSTU"; do
                for TGT in $TGTS; do
                  python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method "RoTTA" --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                    --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum "0.05" \
                    --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                    --loss_scaler ${loss_scaler} \
                    ${validation} \
                    2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                  i=$((i + 1))
                  wait_n
                done
              done
            done
          elif [ "${METHOD}" = "BN_Stats" ]; then
            EPOCH=1
            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do

                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "TENT" ]; then
            EPOCH=1
            lr=0.001
            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do

                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                # python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                #   --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                #   --weight_decay ${weight_decay} \
                #   --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                #   ${validation} \
                #   2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "CoTTA" ]; then
            lr=0.001
            EPOCH=1
            aug_threshold=0.1

            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do

                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --aug_threshold ${aug_threshold} \
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "SAR" ]; then
            EPOCH=1
            lr=0.00025 # From SAR paper: args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025

            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do
                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "EATA" ] || [ "${METHOD}" = "ETA" ]; then
            EPOCH=1

            if [ "${DATASET}" = "pacs" ] ; then
              lr=0.001
              e_margin=0.7784 # 0.4*ln(7)
              d_margin=0.5
              fisher_alpha=2000
            elif [ "${DATASET}" = "tiny-imagenet" ] ; then
              lr=0.001
              e_margin=2.1193 # 0.4*ln(5)
              d_margin=0.5
              fisher_alpha=2000
            fi

            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do
                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                  --lr ${lr} --weight_decay ${weight_decay} \
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}" \
                  --e_margin ${e_margin} --d_margin ${d_margin} --fisher_alpha ${fisher_alpha} \
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "SimATTA" ]; then
            EPOCH=10
            update_every_x="64"
            memory_size="64"

            lr=0.001
            if [ "${DATASET}" = "tiny-imagenet" ]; then
              lr=0.0001
            fi

            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do
                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --use_learned_stats --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type "FIFO"\
                  --lr ${lr} --weight_decay ${weight_decay} --early_stop\
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}"\
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "SimATTA_BIN" ]; then
            EPOCH=10
            update_every_x="64"
            memory_size="64"

            lr=0.001
            if [ "${DATASET}" = "tiny-imagenet" ]; then
              lr=0.0001
            fi

            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do
                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --use_learned_stats --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type "FIFO"\
                  --lr ${lr} --weight_decay ${weight_decay} --early_stop\
                  --log_prefix "${LOG_PREFIX}_${SEED}_dist${dist}"\
                  ${validation} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

                i=$((i + 1))
                wait_n
              done
            done
          elif [ "${METHOD}" = "BiTTA" ] ; then
            restoration_factor=0.0
            if [ "${DATASET}" = "pacs" ]; then
              lr=0.001
              EPOCH=3
              dropout_rate=0.3
            elif [ "${DATASET}" = "cifar10" ]; then
              lr=0.0001
              EPOCH=3
              dropout_rate=0.3
            elif [ "${DATASET}" = "cifar100" ]; then
              lr=0.0001
              EPOCH=3
              dropout_rate=0.3
            elif [ "${DATASET}" = "tiny-imagenet" ] ; then
              lr=0.00005
              EPOCH=5
              dropout_rate=0.1
              restoration_factor=0.01
            fi
            memory_type="ActivePriorityFIFO"
            #### Train with BN
            for dist in "${DISTS[@]}"; do
              for TGT in $TGTS; do

                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL ${CP} --epoch $EPOCH --seed $SEED \
                  --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type}\
                  --weight_decay ${weight_decay} --lr ${lr} --restoration_factor ${restoration_factor}\
                  --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} --use_learned_stats --bn_momentum 0.3 --dropout_rate ${dropout_rate}\
                  ${validation} --sample_selection mc_conf\
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          
                i=$((i + 1))
                wait_n
              done
            done
          fi

        done
      done
    done
  done

  wait
}

test_time_adaptation
