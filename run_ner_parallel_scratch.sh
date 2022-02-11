#!/bin/bash

DEBUG=0
if [[ $DEBUG = 1 ]]; then
    export WANDB_PROJECT=tmp
    export WANDB_ENTITY="murahari"
    OUTPUT_DIR_BASE="checkpoints/debug_random_encoding-ner"
    TASK_NAMES=("ner")
    DATASETS=("conll2003")
    SAVE_STEPS=50000
else
    export WANDB_ENTITY="princeton-nlp"
    export WANDB_PROJECT=pretraining_insights
    OUTPUT_DIR_BASE="checkpoints/parallel-scratch-ner"
    TASK_NAMES=("ner")
    DATASETS=("conll2003")
    SAVE_STEPS=50000
fi

RETRIEVAL_LOSS_COEFF=1
TASK_LOSS_COEFF=1
SENTENCE_CLASSIFICATION_LOSS=0
INFERENCE_TYPE="parallel_scratch_conditional"
RANDOM_ENCODING_NORM=1
BINARY_ENCODING_EPSILON=0
LEARNING_RATE=5e-5

DATALOADER_DROP_LAST=0
RETRIEVAL_PERCENTAGE=1.0
LEARNT_EMBEDDINGS=0

MODEL_PATH="dummy"
CONFIG_PATH="configs/roberta.json"

NUM_SENTENCES_LIST=(2 10 20 40)
# NUM_SENTENCES_LIST=(5)

# LOSS_TYPE_LIST=("mlm_multisentence_binary" "mlm_multisentence_random_orthogonal" "mlm_multisentence_sinusoidal_morespread")
LOSS_TYPE_LIST=("mlm_multisentence_random")

for NUM_SENTENCES in ${NUM_SENTENCES_LIST[@]}; do
    for LOSS_TYPE in ${LOSS_TYPE_LIST[@]}; do
        for TASK_NAME_ID in "${!TASK_NAMES[@]}"; do
            TASK_NAME=${TASK_NAMES[$TASK_NAME_ID]}
            DATASET=${DATASETS[$TASK_NAME_ID]}
            MAX_SEQ_LENGTH=128
            if [[ $NUM_SENTENCES -ge 100 ]]
            then
                MAX_SEQ_LENGTH=$(($NUM_SENTENCES+$MAX_SEQ_LENGTH))
                BATCH_SIZE=6
            elif [[ $NUM_SENTENCES -ge 40 ]]
            then
                BATCH_SIZE=20
            elif [[ $NUM_SENTENCES -ge 20 ]]
            then
                BATCH_SIZE=24
            else
                BATCH_SIZE=28
            fi
            
            if [[ "$INFERENCE_TYPE" = *"parallel"* ]]; then
                BATCH_SIZE=$(($BATCH_SIZE * NUM_SENTENCES))
                DATALOADER_DROP_LAST=1
            fi
            
            if [[ $LEARNT_EMBEDDINGS -ge 1 ]]; then
                OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_learntembed
                RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${NUM_SENTENCES}_${LOSS_TYPE}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_learntembed
            else
                OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}
                RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${NUM_SENTENCES}_${LOSS_TYPE}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}
            fi            

            CMD="python run_ner.py \
            --model_name_or_path ${MODEL_PATH} \
            --tokenizer_name roberta-base \
            --config_name ${CONFIG_PATH} \
            --task_name $TASK_NAME \
            --dataset_name $DATASET \
            --do_train \
            --do_eval \
            --max_seq_length $MAX_SEQ_LENGTH \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --max_steps 500000 \
            --output_dir $OUTPUT_DIR \
            --run_name  $RUN_NAME \
            --overwrite_output_dir \
            --logging_steps 100 \
            --evaluation_strategy steps \
            --eval_steps 5000 \
            --save_steps ${SAVE_STEPS} \
            --sentence_classification_loss $SENTENCE_CLASSIFICATION_LOSS \
            --sentence_loss_fct $LOSS_TYPE \
            --num_sentences $NUM_SENTENCES \
            --inference_type $INFERENCE_TYPE \
            --report_to wandb \
            --dataloader_drop_last $DATALOADER_DROP_LAST \
            --retrieval_percentage $RETRIEVAL_PERCENTAGE \
            --random_encoding_norm $RANDOM_ENCODING_NORM \
            --binary_encoding_epsilon $BINARY_ENCODING_EPSILON \
            --retrieval_loss_coeff $RETRIEVAL_LOSS_COEFF \
            --learnt_embeddings $LEARNT_EMBEDDINGS \
            --task_loss_coeff $TASK_LOSS_COEFF \
            --save_total_limit 10"
            
            if [[ $NUM_SENTENCES -ge 40 ]]
            then
                TIME="120:00:00"
            elif [[ $NUM_SENTENCES -ge 20 ]]
            then
                TIME="72:00:00"
            else
                TIME="30:00:00"
            fi
            
            if [[ $DEBUG = 1 ]]; then
                ./run_job.sh "$CMD"
                # sbatch -A pnlp --time=00:30:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 ./run_job.sh \
                # "$CMD" || scontrol requeue $SLURM_JOB_ID
            else
                sbatch -A pnlp --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 ./run_job.sh \
                "$CMD"
                # sbatch -A pnlp --time=06:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 ./run_job.sh \
                # "$CMD"

            fi
            
        done
    done
done