## DataMUX ##

PyTorch implementation for the paper:

**[DataMUX: Data Multiplexing for Neural Networks](https://arxiv.org/abs/1912.02379)**  
[Vishvak Murahari](https://vishvakmurahari.com/), [Carlos E. Jimenez](https://www.carlosejimenez.com/), [Runzhe Yang](https://runzhe-yang.science/), [Karthik Narasimhan](https://www.cs.princeton.edu/~karthikn/)

![models](images/multiplexing.gif)

This repository contains code for reproducing results. We provide pretrained model weights and associated configs to run inference or train these models from scratch. If you find this work useful in your research, please cite:

```
@article{datamux
  title={DataMUX: Data Multiplexing for Neural Networks,
  author={Vishvak Murahari and Carlos E. Jimenez and Runzhe Yang and Karthik Narasimhan},
  journal={arXiv preprint arXiv:TODO},
  year={2022},
}
```

### Table of Contents

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Usage](#usage)
      * [Overview](#Overview)
      * [Pre-trained checkpoints](#pre-trained-checkpoints)
      * [Training settings](#settings)
      * [Vision Tasks](#vision)
   * [Reference](#reference)
   * [License](#license)

### Setup and Dependencies

Our code is implemented in PyTorch. To setup, do the following:

1. Install [Python 3.6](https://www.python.org/downloads/release/python-365/)
2. Get the source:
```
git clone https://github.com/princeton-nlp/DataMUX.git datamux
```
3. Install requirements into the `datamux` virtual environment, using [Anaconda](https://anaconda.org/anaconda/python):
```
conda env create -f env.yaml
```

### Usage

#### Overview
For sentence-level classification tasks, refer to `run_glue.py` and `run_glue.sh`. For token-level classification tasks, refer to `run_glue.py` and `run_glue.sh`.
#### Pre-trained checkpoints
We release all the pretrained checkpoints on the Hugging Face [model hub](https://huggingface.co/princeton-nlp). We list the checkpoints below. For number of instances, use 2, 5, 10, 20 or 40.

| Task            | Model name on hub | Full path |
| ----------------|:-------------------|---------:
| Retrieval Warmup| datamux-retrieval-<num_instances> | princeton-nlp/datamux-retrieval-<num_instances>|
| MNLI            | datamux-mnli-<num_instances>      | princeton-nlp/datamux-mnli-<num_instances>|
| QNLI            | datamux-qnli-<num_instances>      | princeton-nlp/datamux-qnli-<num_instances>|
| QQP             | datamux-qqp-<num_instances>       | princeton-nlp/datamux-qqp-<num_instances>|
| SST2            | datamux-sst2-<num_instances>      | princeton-nlp/datamux-sst2-<num_instances>|
#### Settings
The bash scripts `run_ner.sh` and `run_glue.sh` take the following arguments:


| Argument      | Flag | Explanation                  |Argument Choices |
| ------------- |:-----|-----------------------------:|-----------------|
| NUM_INSTANCES | -N | Number of multiplexing instances | 2,5,10,20,40 |
| DEMUXING      | -d | Demultiplexing architecture| "index", "mlp" 
| MUXING        | -m | Multiplexing architecture | "gaussian_hadamard", "binary_hadamard", "random_ortho"|
| SETTING       | -m | Training setting | "baseline", "finetuning", "retrieval_pretraining"|
| CONFIG_PATH   | -d | Config path for backbone Transformer Model| Any config file in `configs` directory
| LEARNING_RATE | -m | Learning rate for optimization| Any float but we use either 2e-5 or 5e-5|
| TASK_NAME     | -d | Task name during finetuning| "mnli", "qnli", "sst2", "qqp" for `run_glue.py` or "ner" for `run_ner.py` 
| LEARN_MUXING  | -m | Whether to learn instance embeddings in multiplexing| 0(fixed) or 1(learned)|
| MODEL_PATH    | -d | Model path if either continuing to train from a checkpoint or initialize from retrieval task pretrained checkpoint| Path to local checkpoint or path to model on the [hub](https://huggingface.co/princeton-nlp)
| CONTINUE_TRAIN| -m | Supply flag if resuming training | 0 or 1 (Tnis flag is a little nuanced, please read the sections below for more information)|

Below we list exemplar commands for different training settings:

#### Retrieval pretraining
This commands runs retrieval pretraining for N=2
```
sh run_glue.sh -N 2 -d index -m gaussian_hadamard -s retrieval_pretraining -c configs/ablations/base_model/roberta.json -l 5e-5
```

#### Finetuning
This command finetunes from a retrieval pretrained checkpoint with N=2
```
sh run_glue.sh -N 2 -d index -m gaussian_hadamard -s finetuning -c configs/ablations/base_model/roberta.json -l 5e-5 -t mnli -g "princeton-nlp/datamux-retrieval-2" -k 0
```
Note that we do **not** set -k to 1 as we are not continuing training. For instance, if you wanted to finetune our released checkpoints for more iterations, you could run
```
sh run_glue.sh -N 2 -d index -m gaussian_hadamard -s finetuning -c configs/ablations/base_model/roberta.json -l 5e-5 -t mnli -g "princeton-nlp/datamux-mnli-2" -k 1
```
Note that here we do set -k to 1 as we are continuing training from a checkpoint.

Similar, to run token-level classification tasks like NER, change `run_glue.py` to `run_ner.py`
```
sh run_glue.sh -N 2 -d index -m gaussian_hadamard -s finetuning -c configs/ablations/base_model/roberta.json -l 5e-5 -t mnli -g "princeton-nlp/datamux-retrieval-2" -k 0
```

#### Baselines
For the non-multiplexed baselines, run the following commnands
```
sh run_glue.sh -N 1 -s baseline -c configs/ablations/base_model/roberta.json -l 2e-5 -t mnli
```

#### Vision
For reproducing results on the vision tasks for MLPs and CNNs, please use this [notebook](https://github.com/princeton-nlp/DataMUX/blob/main/vision/vision_multiplexing.ipynb)

### Reference
```
@article{datamux
  title={DataMUX: Data Multiplexing for Neural Networks,
  author={Vishvak Murahari and Carlos E. Jimenez and Runzhe Yang and Karthik Narasimhan},
  journal={arXiv preprint arXiv:TODO},
  year={2022},
}
```
### License
