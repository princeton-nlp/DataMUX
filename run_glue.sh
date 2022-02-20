#!/bin/bash

# flag to run with slurm commands
USE_SLURM=0

# defaults

NUM_INSTANCES=1
DEMUXING="index"
MUXING="gaussian_hadamard"
CONFIG_NAME="configs/ablations/base_model/roberta.json"
LEARNING_RATE=5e-5
TASK_NAME="mnli"
LEARN_MUXING=0
CONTINUE_TRAIN=0
DO_TRAIN=0
DO_EVAL=0
# commmand line arguments
#!/bin/bash

show_help() {
    echo 'Usage run_glue.sh [OPTIONS]'
    echo 'options:'
    echo '-N --num_instances [2,5,10,20,40]'
    echo '-d --demuxing [index, mlp]'
    echo '-m --muxing [gaussian_hadamard, binary_hadamard, random_ortho]'
    echo '-s --setting [baseline, finetuning, retrieval_pretraining]'
    echo '--task [mnli, qnli, sst2, qqp]'
    echo '--config_name CONFIG_NAME'
    echo '--lr LR'
    echo '--batch_size BATCH_SIZE'
    echo '--model_path MODEL_PATH'
    echo '--learn_muxing'
    echo '--continue'
    echo '--do_train'
    echo '--do_eval'
}

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
        ;;
        
        -N|--num_instances)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                NUM_INSTANCES=$2
                shift             # shift consumes $2 without treating it as another argument
            else
                die 'ERROR: "--num-instances" requires a non-empty option argument.'
            fi
        ;;
        
        -d|--demuxing)
            if [ "$2" ]; then
                DEMUXING=$2
                shift
            else
                die 'ERROR: "--demuxing" requires a non-empty option argument.'
            fi
        ;;
        
        -m|--muxing)
            if [ "$2" ]; then
                MUXING=$2
                shift
            else
                die 'ERROR: "--muxing" requires a non-empty option argument.'
            fi
        ;;
        
        -s|--setting)
            if [ "$2" ]; then
                SETTING=$2
                shift
            else
                die 'ERROR: "--setting" requires a non-empty option argument.'
            fi
        ;;
        
        --config_name)
            if [ "$2" ]; then
                CONFIG_NAME=$2
                shift
            else
                die 'ERROR: "--config_name" requires a non-empty option argument.'
            fi
        ;;
        
        --lr)
            if [ "$2" ]; then
                LEARNING_RATE=$2
                shift
            else
                die 'ERROR: "--lr" requires a non-empty option argument.'
            fi
        ;;
        
        --batch_size)
            if [ "$2" ]; then
                BATCH_SIZE=$2
                shift
            else
                die 'ERROR: "--batch_size" requires a non-empty option argument.'
            fi
        ;;
        
        --task)
            if [ "$2" ]; then
                TASK_NAME=$2
                shift
            else
                die 'ERROR: "--task" requires a non-empty option argument.'
            fi
        ;;
        
        --model_path)
            if [ "$2" ]; then
                MODEL_PATH=$2
                shift
            else
                die 'ERROR: "--model_path" requires a non-empty option argument.'
            fi
        ;;
        
        --learn_muxing)
            LEARN_MUXING=1
        ;;
        
        --do_train)
            DO_TRAIN=1
        ;;
        
        --do_eval)
            DO_EVAL=1
        ;;
        
        --)              # End of all options.
            shift
            break
        ;;
        -?*)
            die "ERROR: Unknown option : ${1}"
        ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac
    
    shift
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
    
    # params diff
    # add task name
    # save steps + save strategy + num epochs
    
    CMD_DIFF="--task_name ${TASK_NAME}\
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --max_steps 500000 \
    --save_steps 10000 "

elif [ "$SETTING" = "baseline" ]; then

    echo "Setting is baseline; sets --num-instances to 1."
    RANDOM_ENCODING_NORM=1
    RETRIEVAL_PERCENTAGE=1.0
    RETRIEVAL_PRETRAINING=0
    RETRIEVAL_LOSS_COEFF=0
    TASK_LOSS_COEFF=1
    SHOULD_MUX=0
    DATALOADER_DROP_LAST=0
    OUTPUT_DIR_BASE="checkpoints/baselines"
    NUM_INSTANCES=1
    # add task name
    # save steps + save strategy + num epochs
    CMD_DIFF="--task_name ${TASK_NAME}\
    --evaluation_strategy epoch \
    --num_train_epochs 10"
else
    echo "setting (${SETTING}) not recognized or unset. run \"run_glue.sh -h\" for usage."
    exit 0
fi


if [[ $LEARN_MUXING -ge 1 ]]; then
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_NAME}_learntmuxing
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_NAME}_learnmuxing
else
    OUTPUT_DIR=$OUTPUT_DIR_BASE/${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_NAME}
    RUN_NAME=${TASK_NAME}_${MODEL_PATH}_${MUXING}_${DEMUXING}_${NUM_INSTANCES}_${RETRIEVAL_PERCENTAGE}_norm_${RANDOM_ENCODING_NORM}_rc_${RETRIEVAL_LOSS_COEFF}_lr${LEARNING_RATE}_tc_${TASK_LOSS_COEFF}_${CONFIG_NAME}
fi

if [ -z "$BATCH_SIZE" ]  # if BATCH_SIZE is not set manually
then
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
fi


BATCH_SIZE=$(($BATCH_SIZE * NUM_INSTANCES))

CMD="python run_glue.py \
--tokenizer_name roberta-base \
--config_name ${CONFIG_NAME} \
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
--learn_muxing ${LEARN_MUXING}"

if [ "$DO_TRAIN" -eq 1 ]; then
    CMD="${CMD} --do_train"
fi
if [ "$DO_EVAL" -eq 1 ]; then
    CMD="${CMD} --do_eval"
fi

if [ ! -z "$MODEL_PATH" ]  # if MODEL PATH is set manually
then
    CMD="${CMD} --model_name_or_path ${MODEL_PATH}"
fi

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

echo "Running command with arguments:"
echo $CMD

if [[ $USE_SLURM = 1 ]]; then
    sbatch --time=$TIME --mem=32G --output=logs/%x-%j.out --job-name=${TASK_NAME}_${NUM_INSTANCES}_${MUXING}_${DEMUXING} --gres=gpu:1 ./run_job.sh \
    "$CMD"
else
    ./run_job.sh "$CMD"
fi
