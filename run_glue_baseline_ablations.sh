#!/bin/bash
ONE_HR_JOB_TRICK=0
if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
    SAVE_STEPS=1000
else
    SAVE_STEPS=10000
fi

export WANDB_ENTITY="princeton-nlp"
export WANDB_PROJECT=tmux-icml

OUTPUT_DIR="checkpoints/roberta-scratch"
TASK_NAMES=("qqp" "sst2" "qnli")
# TASK_NAMES=("mnli")

# model settings
NUM_SENTENCES=1
SENTENCE_CLASSIFICATION_LOSS=0

MODEL_PATH="dummy"
INFERENCE_TYPE="baseline_scratch"

BATCH_SIZE=32
DATALOADER_DROP_LAST=0
LEARNING_RATE=2e-5

# these params not used for baseline models, setting a dummy value of 0.5 here
RETRIEVAL_PERCENTAGE=1
LOSS_TYPE="mlm_multisentence_sinusoidal_morespread"

echo $BATCH_SIZE
NUM_EPOCHS=25

CONFIG_ROOT="configs/ablations/"
# CONFIG_DIRS=("base_model" "varying_attention_heads" "varying_width" "varying_depth")
CONFIG_DIRS=("base_model")

for TASK_NAME in ${TASK_NAMES[@]}; do
    for CONFIG_DIR in ${CONFIG_DIRS[@]}; do
        ALL_CONFIGS=$(ls ${CONFIG_ROOT}${CONFIG_DIR})
        for CONFIG in ${ALL_CONFIGS[@]}; do
            CMD="python run_glue.py \
            --model_name_or_path $MODEL_PATH \
            --config_name ${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG} \
            --tokenizer_name roberta-base \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs ${NUM_EPOCHS} \
            --save_total_limit 5 \
            --output_dir $OUTPUT_DIR/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_ablation_${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG}_${LEARNING_RATE}_${NUM_EPOCHS} \
            --run_name ${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_ablation_${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG}_${LEARNING_RATE}_${NUM_EPOCHS} \
            --evaluation_strategy epoch \
            --save_steps $SAVE_STEPS \
            --sentence_classification_loss $SENTENCE_CLASSIFICATION_LOSS \
            --sentence_loss_fct $LOSS_TYPE \
            --num_sentences $NUM_SENTENCES \
            --inference_type $INFERENCE_TYPE \
            --report_to wandb \
            --dataloader_drop_last $DATALOADER_DROP_LAST \
            --retrieval_percentage $RETRIEVAL_PERCENTAGE"
            # ./run_job.sh "$CMD"
            if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
                RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
                echo $RUN_ID
                sbatch -A pnlp --time=00:59:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:rtx_2080:1 --exclude=node000,node001,node016,node916 ./run_job_requeue.sh "$CMD" 59 30 $RUN_ID
            else
                sbatch -A pnlp --time=30:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_ablation --gres=gpu:1 -x node004,node005 ./run_job.sh \
                    "$CMD"
            fi

        done
    done
done
