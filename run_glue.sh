#!/bin/bash

# flag to run with slurm commands
USE_SLURM=1

# search for the latest checkpoint in the directory
get_last_checkpoint_in_output(){
    cur_dir=$(pwd)
    dir=$1
    cd $dir
    array=(*)
    checkpoint_ids=()
    for checkpoint in "${array[@]}";
        do
            IFS='-'
            read -ra ADDR <<< "${checkpoint}";
            checkpoint_ids+=(${ADDR[1]})
        done

    IFS=' '
    max=${checkpoint_ids[0]}
    for n in "${checkpoint_ids[@]}" ; do
        ((n > max)) && max=$n
    done
    last_checkpoint=$max
    cd $cur_dir    
}

# commmand line arguments
while getopts u:a:f: flag
do
    case "${flag}" in
        N) NUM_INSTANCES=${OPTARG};;
        d) DEMUXING=${OPTARG};;
        m) MUXING=${OPTARG};;
        s) SETTING=${OPTARG};;
        c) CONFIG=${OPTARG};;
        lr) LEARNING_RATE=${OPTARG};;
        t) TASK_NAME=${OPTARG};;
        learn_muxing) LEARN_MUXING=${OPTARG};;
        retrieval_checkpoint) RETRIEVAL_CHECKPOINT=${OPTARG};;
    esac
done

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

elif [ "$SETTING" = "finetuning" ]; then

    RANDOM_ENCODING_NORM=1
    RETRIEVAL_PERCENTAGE=1.0
    RETRIEVAL_PRETRAINING=0
    RETRIEVAL_LOSS_COEFF=0.1
    TASK_LOSS_COEFF=0.9
    SHOULD_MUX=1
    DATALOADER_DROP_LAST=1
    OUTPUT_DIR_BASE="checkpoints/finetune"

    # load retrieval warmup checkpoint
    get_last_checkpoint_in_output ${RETRIEVAL_CHECKPOINT}
    MODEL_PATH=${RETRIEVAL_CHECKPOINT}/checkpoint-${last_checkpoint}
    echo $MODEL_PATH

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

else
    echo "setting not recognized, gg"
    exit 0
fi

# get the last checkpoint in the directory
get_last_checkpoint_in_output ${MODEL_CHECKPOINT_DIR}
MODEL_PATH=${MODEL_CHECKPOINT_DIR}/checkpoint-${last_checkpoint}
echo $MODEL_PATH


if [[ $LEARN_MUXING -ge 1 ]]; then
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learntmuxing
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learnmuxing
else
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
fi

if [[ $NUM_SENTENCES -ge 40 ]]
then
    BATCH_SIZE=16

elif [[ $NUM_SENTENCES -ge 20 ]]
then
    BATCH_SIZE=20
elif [[ $NUM_SENTENCES -ge 2 ]]
    BATCH_SIZE=24
else
    BATCH_SIZE=32
fi

BATCH_SIZE=$(($BATCH_SIZE * NUM_INSTANCES))

CMD="python run_glue.py \
--model_name_or_path ${MODEL_PATH} \
--tokenizer_name roberta-base \
--config_name ${CONFIG_PATH} \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length $MAX_SEQ_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--max_steps 500000 \
--output_dir $OUTPUT_DIR \
--run_name  $RUN_NAME \
--logging_steps 100 \
--evaluation_strategy steps \
--eval_steps 10000 \
--save_steps ${SAVE_STEPS} \
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
--learn_muxing ${LEARN_MUXING}"

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
    sbatch --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 ./run_job.sh \
    "$CMD"                
else
    ./run_job.sh "$CMD"
fi
