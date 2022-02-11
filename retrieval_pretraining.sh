DEBUG=0
ONE_HR_JOB_TRICK=0
if [[ $DEBUG = 1 ]]; then
    EVAL_STEPS=300
else
    EVAL_STEPS=3000
fi

if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
    SAVE_STEPS=1000
else
    SAVE_STEPS=50000
fi

export WANDB_PROJECT=tmp
export WANDB_ENTITY="murahari"
OUTPUT_DIR_BASE="checkpoints/debug_random_encoding"
RETRIEVAL_PRETRAINING=1

RETRIEVAL_LOSS_COEFF=1
TASK_LOSS_COEFF=0
SENTENCE_CLASSIFICATION_LOSS=0
INFERENCE_TYPE="parallel_scratch_conditional"
RANDOM_ENCODING_NORM=20
BINARY_ENCODING_EPSILON=0
LEARNING_RATE=2e-5

DATALOADER_DROP_LAST=0
RETRIEVAL_PERCENTAGE=1.0
LEARNT_EMBEDDINGS=0

MODEL_PATH="dummy"
# CONFIG_PATH="configs/roberta.json"
# CONFIG_PATH="configs/ablations/varying_attention_heads/roberta_2heads.json"
# CONFIG_PATH="configs/ablations/varying_width/roberta_384.json"
CONFIG_PATH="configs/ablations/varying_depth/roberta_4layers.json"

# LOSS_TYPE_LIST=("mlm_multisentence_binary" "mlm_multisentence_random_orthogonal" "mlm_multisentence_sinusoidal_morespread")
# NUM_SENTENCES_LIST=(40)
# LOSS_TYPE_LIST=("mlm_multisentence_random" "mlm_multisentence_random_orthogonal")

# NUM_SENTENCES_LIST=(2 5 20)
LOSS_TYPE_LIST=("mlm_multisentence_random")
NUM_SENTENCES_LIST=(1)
# LOSS_TYPE_LIST=("mlm_multisentence_random_orthogonal")

for NUM_SENTENCES in ${NUM_SENTENCES_LIST[@]}; do
    for LOSS_TYPE in ${LOSS_TYPE_LIST[@]}; do
        GRAD_ACC_STEPS=1
        MAX_SEQ_LENGTH=128
        BATCH_SIZE=28
        if [[ $NUM_SENTENCES -ge 100 ]]
        then
            BATCH_SIZE=6
            GRAD_ACC_STEPS=4
        elif [[ $NUM_SENTENCES -ge 40 ]]
        then
            BATCH_SIZE=16
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
        
        OUTPUT_DIR=$OUTPUT_DIR_BASE/${MODEL_PATH}_${INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
        RUN_NAME=${MODEL_PATH}_${INFERENCE_TYPE}_${NUM_SENTENCES}_${LOSS_TYPE}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}

        if [[ $LEARNT_EMBEDDINGS -ge 1 ]]; then
            OUTPUT_DIR=${OUTPUT_DIR}_learntembed
            RUN_NAME=${RUN_NAME}_learntembed
        fi

        CMD="python run_glue.py \
        --model_name_or_path ${MODEL_PATH} \
        --tokenizer_name roberta-base \
        --config_name ${CONFIG_PATH} \
        --do_train \
        --do_eval \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size  $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_steps 500000 \
        --output_dir $OUTPUT_DIR \
        --run_name  $RUN_NAME \
        --logging_steps 100 \
        --evaluation_strategy steps \
        --eval_steps $EVAL_STEPS \
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
        --save_total_limit 5 \
        --retrieval_pretraining ${RETRIEVAL_PRETRAINING} \
        --dataset_name wikitext \
        --dataset_config_name wikitext-103-raw-v1 \
        --gradient_accumulation_steps ${GRAD_ACC_STEPS}"

        
        if [[ $NUM_SENTENCES -ge 40 ]]
        then
            TIME="48:00:00"
            TOT_SLOTS=48
        elif [[ $NUM_SENTENCES -ge 20 ]]
        then
            TIME="30:00:00"
            TOT_SLOTS=30
        else
            TIME="24:00:00"
            TOT_SLOTS=24
        fi
        
        if [[ $DEBUG = 1 ]]; then
            ./run_job.sh "$CMD"
        else
            if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
                RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
                echo $RUN_ID
                sbatch -A pnlp --time=1:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:rtx_3090:1 --exclude=node000,node001,node016,node916 ./run_job_requeue.sh "$CMD" 58 $TOT_SLOTS $RUN_ID
            else
                sbatch -A pnlp --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 -x node004,node005 ./run_job.sh \
                "$CMD"
            fi
        fi
    done
done