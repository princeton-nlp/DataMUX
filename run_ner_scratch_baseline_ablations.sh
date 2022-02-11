#!/bin/bash

export WANDB_ENTITY="princeton-nlp"
export WANDB_PROJECT=pretraining_insights

OUTPUT_DIR_BASE="checkpoints/scratch-ner-pos"
TASK_NAMES=("ner")
DATASETS=("conll2003")

# model settings
NUM_SENTENCES=1
SENTENCE_CLASSIFICATION_LOSS=0

MODEL_PATH="dummy"
INFERENCE_TYPE="baseline_scratch"

BATCH_SIZE=32
DATALOADER_DROP_LAST=0
LEARNING_RATE=5e-5

# these params not used for baseline models, setting a dummy value of 0.5 here
RETRIEVAL_PERCENTAGE=1
LOSS_TYPE="mlm_multisentence_sinusoidal_morespread"

echo $BATCH_SIZE

NUM_EPOCHS=25

CONFIG_ROOT="configs/ablations/"
CONFIG_DIRS=("base_model" "varying_attention_heads" "varying_width" "varying_depth")

for TASK_NAME_ID in "${!TASK_NAMES[@]}"; do
    TASK_NAME=${TASK_NAMES[$TASK_NAME_ID]}
    DATASET=${DATASETS[$TASK_NAME_ID]}
    for CONFIG_DIR in ${CONFIG_DIRS[@]}; do
        ALL_CONFIGS=$(ls ${CONFIG_ROOT}${CONFIG_DIR})
        for CONFIG in ${ALL_CONFIGS[@]}; do
            CMD="python run_ner.py \
            --model_name_or_path $MODEL_PATH \
            --config_name ${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG} \
            --tokenizer_name roberta-base \
            --task_name $TASK_NAME \
            --dataset_name $DATASET \
            --do_train \
            --do_eval \
            --logging_steps 100 \
            --max_seq_length 128 \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs ${NUM_EPOCHS} \
            --output_dir $OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_ablation_${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG}_${LEARNING_RATE}_${NUM_EPOCHS} \
            --run_name ${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_ablation_${CONFIG_ROOT}${CONFIG_DIR}/${CONFIG}_${LEARNING_RATE}_${NUM_EPOCHS} \
            --evaluation_strategy epoch \
            --overwrite_output_dir \
            --save_steps 10000 \
            --sentence_classification_loss $SENTENCE_CLASSIFICATION_LOSS \
            --sentence_loss_fct $LOSS_TYPE \
            --num_sentences $NUM_SENTENCES \
            --inference_type $INFERENCE_TYPE \
            --report_to wandb \
            --dataloader_drop_last $DATALOADER_DROP_LAST \
            --retrieval_percentage $RETRIEVAL_PERCENTAGE"
            # ./run_job.sh "$CMD"
            sbatch -A pnlp --time=24:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_ablation -x node004,node005 --gres=gpu:1 ./run_job.sh \
                "$CMD"
        done
    done
done
