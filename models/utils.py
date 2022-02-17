import torch
import math
import re
import os
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint_trainerstate_robust(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path)) and os.path.exists(os.path.join(folder, path, "trainer_state.json"))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def random_encoding(max_positions, d_model, norm=1):

    gauss = torch.randn((max_positions, d_model))
    gauss = gauss / torch.norm(gauss, dim=1).unsqueeze(1)
    gauss *= norm
    return gauss

def topk(
    logits,
    gt_classes,
    k_list,
):
    assert len(logits.shape) == 2
    assert len(gt_classes.shape) == 1
    batch, _ = logits.shape
    max_k = max(k_list)
    top_labels_max_k = torch.topk(logits, max_k, dim=1)[1]
    return [
        torch.sum(top_labels_max_k[:, :k] == gt_classes.unsqueeze(1)) / batch
        for k in k_list
    ]


def gen_attn_mask(sequence_length, len=None):
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, len)
    seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def binary_encoding(max_position, d_model, epsilon=0.3):
    assert epsilon <= 1 and epsilon >= 0, "epsilon value should lie in [0,1)"
    chunk_size = d_model // max_position
    start_of_chunks = chunk_size * torch.arange(max_position)
    end_of_chunks = start_of_chunks + chunk_size
    end_of_chunks[-1] = d_model
    # tweak start and end states to account for epsilon
    num_intersection = (epsilon / 2) * chunk_size
    start_of_chunks[1:] = start_of_chunks[1:] - num_intersection
    end_of_chunks[:-1] = end_of_chunks[:-1] + num_intersection

    # for loop here :( , not worth vectorizing, only called once
    binary_embeds = torch.zeros(max_position, d_model)
    for pos in range(max_position):
        binary_embeds[pos, start_of_chunks[pos] : end_of_chunks[pos]] = 1
    return binary_embeds
    
def count_params_hf(model):
    params = {k: v for k, v in model.named_parameters()}
    return sum([math.prod(v.shape) for _, v in params.items()])
