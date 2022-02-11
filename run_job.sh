#!/bin/bash
export WANDB_NOTES=$SLURM_JOB_ID
for var in "$@"
do
    $var
done
