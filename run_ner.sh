#!/bin/bash

# flag to run with slurm commands
USE_SLURM=0

# defaults

NUM_INSTANCES=1
DEMUXING="index"
MUXING="gaussian_hadamard"
SETTING="baseline"
CONFIG_PATH="configs/ablations/base_model/roberta.json"
LEARNING_RATE=5e-5
TASK_NAME="ner"
LEARN_MUXING=0
CONTINUE_TRAIN=0
# commmand line arguments
while getopts N:d:m:s:c:l:t:g:k: flag
do
    case "${flag}" in
        N) NUM_INSTANCES=${OPTARG};;
        d) DEMUXING=${OPTARG};;
        m) MUXING=${OPTARG};;
        s) SETTING=${OPTARG};;
        c) CONFIG_PATH=${OPTARG};;
        l) LEARNING_RATE=${OPTARG};;
        t) TASK_NAME=${OPTARG};;
        b) LEARN_MUXING=${OPTARG};;
        g) MODEL_PATH=${OPTARG};;
        k) CONTINUE_TRAIN=${OPTARG};;
    esac
done

declare -A task2datasetmap
task2datasetmap[ner]="conll2003"
DATASET=${task2datasetmap[$TASK_NAME]}
# other miscelleneous params
SAVE_STEPS=10000
MAX_SEQ_LENGTH=128

if [ "$SETTING" == "retrieval_pretraining" ]; then

    RANDOM_ENCODING_NORM=20
    RETRIEVAL_PERCENTAGE=1.0
    RETRIEVAL_PRETRAINING=1
    RETRIEVAL_LOSS_COEFF=1
    TASK_LOSS_COEFF=0
    SHOULD_MUX=1
    DATALOADER_DROP_LAST=1
    OUTPUT_DIR_BASE="checkpoints/retrieval_pretraining"
    MODEL_PATH="retrieval_pretraining"

    # params diff
    DATASET_NAME="wikitext"
    DATASET_CONFIG_NAME="wikitext-103-raw-v1"
    CMD_DIFF="--dataset_name ${DATASET_NAME}\
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --max_steps 500000 \
    --save_steps 10000"

elif [ "$SETTING" = "finetuning" ]; then

    RANDOM_ENCODING_NORM=1
    RETRIEVAL_PERCENTAGE=1.0
    RETRIEVAL_PRETRAINING=0
    RETRIEVAL_LOSS_COEFF=0.1
    TASK_LOSS_COEFF=0.9
    SHOULD_MUX=1
    DATALOADER_DROP_LAST=1
    OUTPUT_DIR_BASE="checkpoints/finetune"

    # add task name
    # save steps + save strategy + num epochs

    CMD_DIFF="--task_name ${TASK_NAME}\
    --dataset_name $DATASET \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --max_steps 500000 \
    --save_steps 10000 "

elif [ "$SETTING" = "baseline" ]; then

    RANDOM_ENCODING_NORM=1
    RETRIEVAL_PERCENTAGE=1.0
    RETRIEVAL_PRETRAINING=0
    RETRIEVAL_LOSS_COEFF=0
    TASK_LOSS_COEFF=1
    SHOULD_MUX=0
    DATALOADER_DROP_LAST=0
    OUTPUT_DIR_BASE="checkpoints/baselines"
    NUM_INSTANCES=1
    MODEL_PATH="baseline"
    # add task name 
    # save steps + save strategy + num epochs
    CMD_DIFF="--task_name ${TASK_NAME}\
    --dataset_name $DATASET \
    --evaluation_strategy epoch \
    --num_train_epochs 10"
else
    echo "setting not recognized, gg"
    exit 0
fi

if [[ $LEARN_MUXING -ge 1 ]]; then
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learntmuxing
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learnmuxing
else
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
fi

if [[ $NUM_INSTANCES -ge 40 ]]
then
    BATCH_SIZE=16

elif [[ $NUM_INSTANCES -ge 20 ]]
then
    BATCH_SIZE=20
elif [[ $NUM_INSTANCES -ge 2 ]]
then
    BATCH_SIZE=24
else
    BATCH_SIZE=32
fi

BATCH_SIZE=$(($BATCH_SIZE * NUM_INSTANCES))

CMD="python run_ner.py \
--model_name_or_path ${MODEL_PATH} \
--tokenizer_name roberta-base \
--config_name ${CONFIG_PATH} \
--do_train \
--do_eval \
--max_seq_length $MAX_SEQ_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--output_dir $OUTPUT_DIR \
--run_name  $RUN_NAME \
--logging_steps 100 \
--report_to wandb \
--dataloader_drop_last $DATALOADER_DROP_LAST \
--retrieval_percentage $RETRIEVAL_PERCENTAGE \
--retrieval_loss_coeff $RETRIEVAL_LOSS_COEFF \
--task_loss_coeff $TASK_LOSS_COEFF \
--retrieval_pretraining ${RETRIEVAL_PRETRAINING} \
--num_instances ${NUM_INSTANCES} \
--muxing_variant ${MUXING} \
--demuxing_variant ${DEMUXING} \
--should_mux ${SHOULD_MUX} \
--gaussian_hadamard_norm ${RANDOM_ENCODING_NORM} \
--learn_muxing ${LEARN_MUXING} \
--continue_train ${CONTINUE_TRAIN}"

CMD=${CMD}" "${CMD_DIFF}

if [[ $NUM_INSTANCES -ge 40 ]]
then
    TIME="120:00:00"
elif [[ $NUM_INSTANCES -ge 20 ]]
then
    TIME="72:00:00"
else
    TIME="30:00:00"
fi

if [[ $USE_SLURM = 1 ]]; then
    sbatch --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_INSTANCES}_${MUXING}_${DEMUXING} --gres=gpu:1 ./run_job.sh \
    "$CMD"                
else
    ./run_job.sh "$CMD"
fi
