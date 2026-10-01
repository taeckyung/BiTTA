  LOG_PREFIX="reproduce_src"

  DATASETS="tiny-imagenet" # cifar10 or cifar100
  METHODS="Src"
  VALIDATIONS=(
    # "--log_name resnet50_pretrained/resnet18/
    # --wandb cifar10_resnet50 --wandb_name test_resnet18"

    # resnet 50
    # "--epoch 20 --lr 0.001 --batch_size 64 --log_name pretrained_test/
    # --wandb binary-tta-tiny --wandb_name ground_pretrain_ep20_lr0001_bs64"

    "--epoch 20 --lr 0.001 --batch_size 64 --log_name pretrained_test/2/
    --wandb binary-tta-tiny --wandb_name ground_pretrain_ep20_lr0001_bs64"

    # vitbase16_pretrained

    # "--epoch 150 --lr 0.01 --batch_size 64 --log_name vit_16_pre_trained/pretrain_ep300_lr001_bs64_patch4/ --vit_patch_size 4
    # --wandb cifar10_vit16 --wandb_name pretrain_ep300_lr001_bs64_patch4"

  )

  
  MAIN_SCRIPT="main.py"


  echo DATASETS: $DATASETS
  echo METHODS: $METHODS
    
  
  GPUS=(0 1 2 3 4) #available gpus
  NUM_GPUS=${#GPUS[@]}

  sleep 1 # prevent mistake
  mkdir raw_logs # save console outputs here

  #### Useful functions
  wait_n() {
    #limit the max number of jobs as NUM_MAX_JOB and wait
    background=($(jobs -p))
    local default_num_jobs=7 #num concurrent jobs
    local num_max_jobs=${1:-$default_num_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
      wait -n
    fi
  }

  ###############################################################
  ##### Source Training; Source Evaluation: Source domains  #####
  ###############################################################
  train_source_model() {
    i=0
    update_every_x="64"
    memory_size="64"
    for DATASET in $DATASETS; do
      for METHOD in $METHODS; do
        for VALIDATION in "${VALIDATIONS[@]}"; do


          if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
            EPOCH=200
            # MODEL="resnet18"
            # MODEL="resnet50_pretrained"
            # MODEL="resnet50"
            MODEL="vitbase16"


            TGT="test"
          elif [ "${DATASET}" = "cifar100" ]; then
            EPOCH=200
            MODEL="resnet18"
            TGT="test"
          elif [ "${DATASET}" = "imagenet" ]; then
            EPOCH=30
            MODEL="resnet18_pretrained"
            TGT="test"
          elif [ "${DATASET}" = "tiny-imagenet" ]; then
            EPOCH=10
            MODEL="resnet18_pretrained"
            TGT="test"
          elif [ "${DATASET}" = "pacs" ]; then
            # EPOCH=60
            MODEL="resnet18_pretrained"
            TGT="test"
          elif [ "${DATASET}" = "vlcs" ]; then
            # EPOCH=60
            MODEL="resnet18_pretrained"
            TGT="test"
          elif [ "${DATASET}" = "office_home" ]; then
            EPOCH=200
            MODEL="resnet50_pretrained"
            TGT="test"
          fi

          for SEED in 0; do
            if [[ "$METHOD" == *"Src"* ]]; then
              #### Train with BN
              for tgt in $TGT; do
                # python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset ${DATASET} --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                #   --log_prefix ${LOG_PREFIX}_${SEED} \
                #   ${VALIDATION} \
                #   2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
                
                python ${MAIN_SCRIPT} --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset ${DATASET} --method Src --tgt ${tgt} --model $MODEL --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                  --log_prefix ${LOG_PREFIX}_${SEED} \
                  ${VALIDATION} \
                  2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &
                i=$((i + 1))
                wait_n
              done
            fi
          done
        done
      done
    done

    wait
  }

  train_source_model