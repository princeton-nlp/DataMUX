from calendar import c
import transformers
from transformers import AutoConfig
import wandb
import os
from models.multiplexing import RobertaSequenceClassificationMuxed
import torch
import subprocess
from datetime import datetime

N = [2, 5, 10, 20, 40]

mnli_checkpoints = [
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/2jgvmcqk",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/2dw7y6ik",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/1pvbrs5l",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/1z4z1hhu",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/240z6el8",
]

qnli_checkpoints = [
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/14wduwwo",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/21k1y9gq",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/16x37myg",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/3nfpj8wl",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/39tpllu0",
]

qqp_checkpoints = [
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/37nfigvs",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/3k0gdl2b",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/2py0ancj",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/1oyovqho",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/u9fwhjei",
]

sst2_checkpoints = [
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/iyrkas2x",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/3megvsw1",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/2x7mmo35",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/2jw42pej",
    "https://wandb.ai/murahari/tmp_retrieval_pretrain_finetune/runs/3ogudisr",
]

TASK_2_CHECKPOINTS = {
    "mnli": mnli_checkpoints,
    "qnli": qnli_checkpoints,
    "qqp": qqp_checkpoints,
    "sst2": sst2_checkpoints,
}
TASK_2_LABELS = {
    "mnli": 3,
    "qnli": 2,
    "qqp": 2,
    "sst2": 2,
}
TASKS = list(TASK_2_CHECKPOINTS.keys())


def get_immediate_subdirectories(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def find_last_checkpoint(
    output_directory, checkpoint_prefix="checkpoint", delimiter="-"
):
    subdirs = get_immediate_subdirectories(output_directory)
    checkpoint_iterations = [
        x.split(delimiter)[1] for x in subdirs if checkpoint_prefix in x
    ]
    checkpoint_iterations = list(map(int, checkpoint_iterations))
    return f"checkpoint-{max(checkpoint_iterations)}"

"""
# iterate through all apis and create indices
url2run = {}
api = wandb.Api()
runs = api.runs("murahari/tmp_retrieval_pretrain_finetune")
for run in runs:
    url2run[run.url] = run

base_dir = os.getcwd()
# iterate through all tasks
for task in TASKS:
    urls = TASK_2_CHECKPOINTS[task]
    for j, url in enumerate(urls):
        run = url2run[url]
        config = run.config
        output_dir = config["output_dir"]
        num_instances = N[j]
        model_name = f"datamux_base_{task}_{N[j]}"
        last_checkpoint = find_last_checkpoint(output_dir)

        config = AutoConfig.from_pretrained(
            "configs/ablations/base_model/roberta.json",
            num_labels=TASK_2_LABELS[task],
            finetuning_task=task,
        )
        config.num_instances = num_instances
        config.muxing_variant = "gaussian_hadamard"
        config.demuxing_variant = "index"
        config.retrieval_percentage = 1
        config.gaussian_hadamard_norm = 1
        config.binary_hadamard_epsilon = 0
        config.retrieval_loss_coeff = 0.1
        config.task_loss_coeff = 0.9
        config.learn_muxing = 0
        # load the right model
        model = RobertaSequenceClassificationMuxed(config)
        # load the checkpoint
        state_dict = torch.load(
            os.path.join(output_dir, last_checkpoint, "pytorch_model.bin"),
            map_location=torch.device("cpu"),
        )
        # massage the state dict to new structure
        del_keys = [k for k in state_dict if "lm_head" in k]
        new_keys = []
        for k in del_keys:
            new_keys.append(".".join(["demultiplexer"] + k.split(".")[1:]))
        del_keys.append("sentence_embedding")
        new_keys.append("instance_embedding")
        assert len(new_keys) == len(del_keys)
        state_dict_updates = {
            new_keys[j]: state_dict[del_keys[j]] for j in range(len(del_keys))
        }
        for k in del_keys:
            state_dict.pop(k)
        state_dict.update(state_dict_updates)
        # rename the demultiplexer module
        model.load_state_dict(state_dict, strict=False)

        # push to hub
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
        tmp_folder = os.path.join("/tmp", date_time)
        # create repo
        model_name = f"datamux-{task}-{num_instances}"
        os.makedirs(os.path.join(tmp_folder, model_name))
        subprocess.run(["huggingface-cli", "repo", "create", model_name])
        # change directory and commit and push stuff
        os.chdir(tmp_folder)
        subprocess.run(
            ["git", "clone", f"https://huggingface.co/princeton-nlp/{model_name}"]
        )
        os.chdir(os.path.join(tmp_folder, model_name))
        model.save_pretrained(os.path.join(tmp_folder, model_name))
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "init commit"])
        subprocess.run(["git", "push"])
        os.chdir(base_dir)
"""

base_dir = os.getcwd()

retrieval_checkpoints = [
    "checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_2_1.0_epsilon_0_norm_20_rc_1_lr5e-5_tc_0_configs/roberta.json/checkpoint-4000",
    "checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_5_1.0_epsilon_0_norm_20_rc_1_lr5e-5_tc_0_configs/roberta.json/checkpoint-166000",
    "checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_10_1.0_epsilon_0_norm_20_rc_1_lr5e-5_tc_0_configs/roberta.json/checkpoint-96000",
    "checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_20_1.0_epsilon_0_norm_20_rc_1_lr5e-5_tc_0_configs/roberta.json/checkpoint-150000",
    "checkpoints/debug_random_encoding/dummy_parallel_scratch_conditional_mlm_multisentence_random_40_1.0_epsilon_0_norm_20_rc_1_lr2e-5_tc_0_configs/roberta.json/checkpoint-359000",
]

for j, checkpoint in enumerate(retrieval_checkpoints):
        num_instances = N[j]
        config = AutoConfig.from_pretrained(
        "configs/ablations/base_model/roberta.json",
        num_labels=2
        )
        config.num_instances = num_instances
        config.muxing_variant = "gaussian_hadamard"
        config.demuxing_variant = "index"
        config.retrieval_percentage = 1
        config.gaussian_hadamard_norm = 20
        config.binary_hadamard_epsilon = 0
        config.retrieval_loss_coeff = 1
        config.task_loss_coeff = 0
        config.learn_muxing = 0
        # load the right model
        model = RobertaSequenceClassificationMuxed(config)
        # load the checkpoint
        state_dict = torch.load(
            os.path.join(checkpoint, "pytorch_model.bin"),
            map_location=torch.device("cpu"),
        )
        # massage the state dict to new structure
        del_keys = [k for k in state_dict if "lm_head" in k]
        new_keys = []
        for k in del_keys:
            new_keys.append(".".join(["demultiplexer"] + k.split(".")[1:]))
        del_keys.append("sentence_embedding")
        new_keys.append("instance_embedding")
        assert len(new_keys) == len(del_keys)
        state_dict_updates = {
            new_keys[j]: state_dict[del_keys[j]] for j in range(len(del_keys))
        }
        for k in del_keys:
            state_dict.pop(k)
        state_dict.update(state_dict_updates)
        # rename the demultiplexer module
        model.load_state_dict(state_dict, strict=False)

        # push to hub
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
        tmp_folder = os.path.join("/tmp", date_time)
        # create repo
        model_name = f"datamux-retrieval-{num_instances}"
        os.makedirs(os.path.join(tmp_folder, model_name))
        subprocess.run(["huggingface-cli", "repo", "create", model_name])
        # change directory and commit and push stuff
        os.chdir(tmp_folder)
        subprocess.run(
            ["git", "clone", f"https://huggingface.co/princeton-nlp/{model_name}"]
        )
        os.chdir(os.path.join(tmp_folder, model_name))
        model.save_pretrained(os.path.join(tmp_folder, model_name))
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "init commit"])
        subprocess.run(["git", "push"])
        os.chdir(base_dir)
