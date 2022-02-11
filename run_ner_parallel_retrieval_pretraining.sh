#!/bin/bash
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

DEBUG=0
ONE_HR_JOB_TRICK=0

TASK_NAMES=("ner")
DATASETS=("conll2003")

if [[ $DEBUG = 1 ]]; then
    export WANDB_PROJECT=tmp_tmp
    export WANDB_ENTITY="murahari"
    OUTPUT_DIR_BASE="checkpoints/debug"
    SAVE_STEPS=10000
else
    export WANDB_ENTITY="murahari"
    export WANDB_PROJECT=tmp_retrieval_pretrain_finetune
    OUTPUT_DIR_BASE="checkpoints/retrieval_pretrain_finetune"

    if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
        SAVE_STEPS=1000
    else
        SAVE_STEPS=50000
    fi
fi

RETRIEVAL_LOSS_COEFF=0.1
TASK_LOSS_COEFF=0.9
SENTENCE_CLASSIFICATION_LOSS=0
INFERENCE_TYPE="parallel_conditional"
RANDOM_ENCODING_NORM=1
BINARY_ENCODING_EPSILON=0
LEARNING_RATE=2e-5

DATALOADER_DROP_LAST=0
RETRIEVAL_PERCENTAGE=1.0
LEARNT_EMBEDDINGS=0

declare -A pretraining_inference_type_map
pretraining_inference_type_map[parallel_conditional]=parallel_scratch_conditional
pretraining_inference_type_map[parallel_mlp]=parallel_scratch_mlp

CONFIG_PATH="configs/ablations/varying_depth/roberta_4layers.json"
# CONFIG_PATH="configs/ablations/varying_width/roberta_384.json"
# CONFIG_PATH="configs/ablations/varying_attention_heads/roberta_2heads.json"

# CONFIG_PATH="configs/roberta.json"
# NUM_SENTENCES_LIST=(5 10 20 40)

# NUM_SENTENCES_LIST=(2 5 10 20 40)
NUM_SENTENCES_LIST=(1)

# LOSS_TYPE_LIST=("mlm_multisentence_binary" "mlm_multisentence_random_orthogonal" "mlm_multisentence_sinusoidal_morespread")
LOSS_TYPE_LIST=("mlm_multisentence_random")

for NUM_SENTENCES in ${NUM_SENTENCES_LIST[@]}; do
    for LOSS_TYPE in ${LOSS_TYPE_LIST[@]}; do
        for TASK_NAME_ID in "${!TASK_NAMES[@]}"; do

            TASK_NAME=${TASK_NAMES[$TASK_NAME_ID]}            
            DATASET=${DATASETS[$TASK_NAME_ID]}

            PRETRAIN_INFERENCE_TYPE=${pretraining_inference_type_map[$INFERENCE_TYPE]}
            echo $PRETRAIN_INFERENCE_TYPE
            # find the checkpoint directory based on the config file
            if [ "$CONFIG_PATH" == "configs/roberta.json" ]; then
                # assign model checkpoint based on number of sentences
                # norm = 20 for random embeddings; norm=1 for orthogonal
                # lr = 2e-5 for N = 40, 2e-5 for N = 20 MLP

                if [ "$LOSS_TYPE" == "mlm_multisentence_random" ]; then
                    PRETRAIN_NORM=20
                elif [ "$LOSS_TYPE" == "mlm_multisentence_random_orthogonal" ]; then
                    PRETRAIN_NORM=1
                elif [ "$LOSS_TYPE" == "mlm_multisentence_binary" ]; then
                    PRETRAIN_NORM=20
                else
                    echo "gg bro"
                    exit 0
                fi

                PRETRAIN_LR=5e-5

                if [ $NUM_SENTENCES -eq 40 ]; then
                    PRETRAIN_LR=2e-5
                fi

                if [ $NUM_SENTENCES -eq 1 ]; then
                    PRETRAIN_LR=2e-5
                fi

                if [ $NUM_SENTENCES -eq 20 ]; then
                    if [ "$PRETRAIN_INFERENCE_TYPE" == "parallel_scratch_mlp" ]; then
                        PRETRAIN_LR=2e-5
                    fi
                fi

                MODEL_CHECKPOINT_DIR="/u/murahari/pretraining_insights/checkpoints/debug_random_encoding/dummy_${PRETRAIN_INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_1.0_epsilon_0_norm_${PRETRAIN_NORM}_rc_1_lr${PRETRAIN_LR}_tc_0_${CONFIG_PATH}"
                MODEL_CHECKPOINT_DIR="/u/murahari/pretraining_insights/checkpoints/debug_random_encoding/dummy_${PRETRAIN_INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_1.0_epsilon_0_norm_${PRETRAIN_NORM}_rc_1_lr${PRETRAIN_LR}_tc_0_${CONFIG_PATH}"
                if [[ $LEARNT_EMBEDDINGS -ge 1 ]]; then
                    MODEL_CHECKPOINT_DIR=${MODEL_CHECKPOINT_DIR}_learntembed
                fi
            
            elif [ "$CONFIG_PATH" = "configs/ablations/varying_width/roberta_384.json" ]; then

                if [ $NUM_SENTENCES -eq 1 ]; then
                                                                            
                    MODEL_CHECKPOINT_DIR="checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_1_1.0_epsilon_0_norm_20_rc_1_lr2e-5_tc_0_configs/ablations/varying_width/roberta_384.json"
                else
                    MODEL_CHECKPOINT_DIR="/u/murahari/pretraining_insights/checkpoints/debug_random_encoding/dummy_${PRETRAIN_INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_1.0_epsilon_0_norm_1_rc_1_lr5e-5_tc_0_${CONFIG_PATH}"
                fi
            elif  [ "$CONFIG_PATH" = "configs/ablations/varying_depth/roberta_4layers.json" ]; then
                # norm = 1 for all
                # lr = 2e -5 for N = 40 MLP; lr = 5e-5;

                PRETRAIN_NORM=1

                PRETRAIN_LR=5e-5

                if [ $NUM_SENTENCES -eq 40 ]; then
                    if [ "$PRETRAIN_INFERENCE_TYPE" == "parallel_scratch_mlp" ]; then
                        PRETRAIN_LR=2e-5
                    fi
                fi

                if [ $NUM_SENTENCES -eq 1 ]; then
                    MODEL_CHECKPOINT_DIR="checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_1_1.0_epsilon_0_norm_20_rc_1_lr2e-5_tc_0_configs/ablations/varying_depth/roberta_4layers.json"
                fi

                
                # MODEL_CHECKPOINT_DIR="/u/murahari/pretraining_insights/checkpoints/debug_random_encoding/dummy_${PRETRAIN_INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_1.0_epsilon_0_norm_1_rc_1_lr5e-5_tc_0_${CONFIG_PATH}"

            elif [ "$CONFIG_PATH" == "configs/ablations/varying_attention_heads/roberta_2heads.json" ]; then

                if [ "$LOSS_TYPE" == "mlm_multisentence_random" ]; then
                    PRETRAIN_NORM=20
                fi
                PRETRAIN_LR=5e-5

                if [ $NUM_SENTENCES -eq 40 ]; then
                    PRETRAIN_LR=2e-5
                fi
                
                MODEL_CHECKPOINT_DIR="/u/murahari/pretraining_insights/checkpoints/debug_random_encoding/dummy_${PRETRAIN_INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_1.0_epsilon_0_norm_${PRETRAIN_NORM}_rc_1_lr${PRETRAIN_LR}_tc_0_${CONFIG_PATH}"
                if [[ $LEARNT_EMBEDDINGS -ge 1 ]]; then
                    MODEL_CHECKPOINT_DIR=${MODEL_CHECKPOINT_DIR}_learntembed
                fi
            
            else
                echo "Config path not recognized, get your life together bro"
                exit 0
            fi

            
            MAX_SEQ_LENGTH=128
            if [[ $NUM_SENTENCES -ge 100 ]]
            then
                MAX_SEQ_LENGTH=$(($NUM_SENTENCES+$MAX_SEQ_LENGTH))
                # BATCH_SIZE=6
            elif [[ $NUM_SENTENCES -ge 40 ]]
            then
                # BATCH_SIZE=10
                # BATCH_SIZE=16
                BATCH_SIZE=8

            elif [[ $NUM_SENTENCES -ge 20 ]]
            then
                # BATCH_SIZE=12
                # BATCH_SIZE=20
                BATCH_SIZE=10

            else
                # BATCH_SIZE=14
                # BATCH_SIZE=24
                BATCH_SIZE=12
            fi
            
            if [[ "$INFERENCE_TYPE" = *"parallel"* ]]; then
                BATCH_SIZE=$(($BATCH_SIZE * NUM_SENTENCES))
                DATALOADER_DROP_LAST=1
            fi

            # get the last checkpoint in the directory
            get_last_checkpoint_in_output ${MODEL_CHECKPOINT_DIR}
            MODEL_PATH=${MODEL_CHECKPOINT_DIR}/checkpoint-${last_checkpoint}
            echo $MODEL_PATH

            if [[ $LEARNT_EMBEDDINGS -ge 1 ]]; then
                OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learntembed
                RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${NUM_SENTENCES}_${LOSS_TYPE}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}_learntembed
            else
                OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${LOSS_TYPE}_${NUM_SENTENCES}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
                RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${INFERENCE_TYPE}_${NUM_SENTENCES}_${LOSS_TYPE}_${RETRIEVAL_PERCENTAGE}_epsilon_${BINARY_ENCODING_EPSILON}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_PATH}
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
            --save_total_limit 5"
            if [[ $NUM_SENTENCES -ge 40 ]]
            then
                TIME="120:00:00"
                TOT_SLOTS=120
            elif [[ $NUM_SENTENCES -ge 20 ]]
            then
                TIME="72:00:00"
                TOT_SLOTS=72
            else
                TIME="30:00:00"
                TOT_SLOTS=30
            fi
            if [[ $DEBUG = 1 ]]; then
                ./run_job.sh "$CMD"

            else
                if [[ $ONE_HR_JOB_TRICK = 1 ]]; then
                    RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
                    echo $RUN_ID
                    sbatch -A pnlp --time=1:00:00 --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:rtx_2080:2 --exclude=node000,node001,node016,node916,node915 ./run_job_requeue.sh "$CMD" 58 $TOT_SLOTS $RUN_ID
                else
                    sbatch -A pnlp --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_SENTENCES}_${INFERENCE_TYPE}_${LOSS_TYPE} --gres=gpu:1 -x node004 ./run_job.sh \
                    "$CMD"
                fi
            fi
        done
    done
done
