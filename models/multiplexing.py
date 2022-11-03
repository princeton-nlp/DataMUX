from re import X
from dataclasses import dataclass

from transformers.utils import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
)
from typing import Optional, Tuple
from transformers.activations import gelu
import math
import numpy as np
from .utils import (
    random_encoding,
    binary_encoding,
)
from scipy.stats import ortho_group, special_ortho_group

logger = logging.get_logger(__name__)

####### SEQUENCE CLASSIFICATION CLASSES

class RobertaSequenceClassificationMuxed(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if config.demuxing_variant == "index":
            self.demultiplexer = RobertaIndexDemultiplexerSequenceClassification(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "mlp":
            self.demultiplexer = RobertaMLPDemultiplexerSequenceClassification(config)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(config)
        else:
            raise NotImplementedError("demuxing_varaint argument (%s) not recognized." % config.demuxing_variant)

        self.init_weights()

        d_model = config.hidden_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError("muxing_variant argument (%s) not recognized." % config.muxing_variant)

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position=None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full((num_instances, num_instances), 50000, device=input_ids.device)
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device)
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + self.roberta.embeddings.word_embeddings.weight.shape[0]
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 49923, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp":
            cls_tokens = torch.full((num_instances, 1), 49923, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            # prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 1
        
        else:
            raise NotImplementedError("demuxing_variant (%s) not recognized." % self.demuxing_variant)

        # concatenate
        embedding_output = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        _, _, embedding_dim = embedding_output.shape
        if self.muxing_variant == "random_ortho":
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )

            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.roberta(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        logits, demuxed_representations = self.demultiplexer(sequence_output)
        if labels is not None:

            labels = labels[: (modified_batch_size * num_instances)]
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position :] = torch.randint(
                num_instances, (modified_batch_size, modified_seq_length - special_tokens_end_position), device=input_ids.device)

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels = torch.div(retrieval_labels, self.config.retrieval_loss_vocab_scale, rounding_mode='trunc')
            retrieval_labels = retrieval_labels.long()
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 1
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = ~pad_mask & torch.bernoulli(
                torch.full(retrieval_labels.shape, 1 - self.config.retrieval_percentage, device=input_ids.device)
            ).bool()
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            retrieval_loss = loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size)
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            hidden_states=demuxed_representations,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
        )

####### TOKEN CLASSIFICATION CLASSES

class RobertaTokenClassificationMuxed(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if config.demuxing_variant == "index":
            self.demultiplexer = RobertaIndexDemultiplexerTokenClassification(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "mlp":
            self.demultiplexer = RobertaMLPDemultiplexerTokenClassification(config)
            self.retrieval_head = RetrievalHeadMLPDemultiplexing(config)
        else:
            raise NotImplementedError("demuxing_variant (%s) not recognized." % config.demuxing_variant)

        self.init_weights()

        d_model = config.hidden_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError("muxing_variant (%s) not recognized." % config.muxing_variant)

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position=None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full((num_instances, num_instances), 50000, device=input_ids.device)
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + self.roberta.embeddings.word_embeddings.weight.shape[0]
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 49923, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "mlp":
            cls_tokens = torch.full((num_instances, 1), 49923, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            # prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 0
        
        else:
            raise NotImplementedError("demuxing_variant (%s) not recognized." % self.demuxing_variant)

        # concatenate
        embedding_output = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        _, _, embedding_dim = embedding_output.shape
        if self.muxing_variant == "random_ortho":
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )

            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.roberta(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        logits, demuxed_representations = self.demultiplexer(sequence_output)
        if labels is not None:
            # retrieval loss calculation
            labels = labels[: (modified_batch_size * num_instances)]
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position :] = torch.randint(
                num_instances, (modified_batch_size, modified_seq_length - special_tokens_end_position), device=input_ids.device
            )

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 1
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = ~pad_mask & torch.bernoulli(
                torch.full(retrieval_labels.shape, 1 - self.config.retrieval_percentage, device=input_ids.device)
            ).bool()
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None
        if labels is not None:
            if attention_mask is not None:
                loss_fct = CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                logits = logits[:, special_tokens_end_position:, :]

                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                task_loss = loss_fct(active_logits, active_labels)
                retrieval_loss = loss_fct(
                    retrieval_predictions.view(-1, self.config.vocab_size),
                    retrieval_labels.view(-1),
                )
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            hidden_states=demuxed_representations,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
        )

####### INDEX DEMUXING CLASSES #########
class RobertaIndexDemultiplexerSequenceClassification(nn.Module):
    """Head for sequence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense_before_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        _, _, _ = features.shape
        positional_embeds = features[
            :, : self.num_instances, :
        ]  # take <s> token (equiv. to [CLS])
        positional_embeds = positional_embeds.reshape(
            -1, positional_embeds.shape[-1]
        )
        # extract the added [CLS] token during inference
        x = features[:, self.num_instances, :]
        x = x.unsqueeze(1).repeat(1, self.num_instances, 1)
        x = x.view(-1, x.shape[-1])

        x = torch.cat([positional_embeds, x], dim=1)
        x = self.dense(x)
        x = gelu(x)
        demuxed_feat = self.layer_norm(x)
        x = self.dense_before_out_proj(demuxed_feat)
        x = gelu(x)
        x = self.out_proj(x)
        return x, demuxed_feat

class RetrievalHeadIndexDemultiplexing(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, math.ceil(config.vocab_size / config.retrieval_loss_vocab_scale))
        self.bias = nn.Parameter(torch.zeros(math.ceil(config.vocab_size / config.retrieval_loss_vocab_scale)))
        self.decoder.bias = self.bias

    def forward(self, features, instance_labels, **kwargs):
        # extract the first <num instance> representations and concatenate with the right word
        batch, seqlength, _ = features.shape
        positional_representations = features[:, : self.num_instances, :]
        # concatenate features with the instance representations based on instance labels
        instance_labels_copy = instance_labels.clone()
        instance_labels_copy[instance_labels == -100] = 0
        positional_embeds = positional_representations[
            torch.arange(batch, device=features.device).unsqueeze(1).repeat(1, seqlength),
            instance_labels_copy,
        ]
        features = torch.cat([positional_embeds, features], dim=2)
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class RobertaIndexDemultiplexerTokenClassification(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.num_labels)
        self.bias = nn.Parameter(torch.zeros(config.num_labels))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):

        # extract the first <num sentence> representations and concatenate with the right word
        batch, seqlength, feature_dim = features.shape
        positional_representations = features[:, : self.num_instances, :]
        # concatenate features with the sentence representations based on sentence_labels
        # don't overwrite sentence labels !!

        # need to expand the batch to the original size, need to make predictions
        # on the original
        positional_representations = positional_representations.unsqueeze(2).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = features.unsqueeze(1).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = torch.cat([positional_representations, features], dim=3)
        # increase the batch size by collapsing the first 2 dimensions
        features = features.view(-1, seqlength, 2 * feature_dim)
        x = self.dense(features)
        x = gelu(x)
        demuxed_feat = self.layer_norm(x)
        x = self.decoder(demuxed_feat)
        return x, demuxed_feat

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

####### MLP DEMUXING CLASSES #########

class RobertaMLPDemuxModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different instances
        for sent_id in range(self.num_instances):
            setattr(
                self,
                f"dense_{sent_id}",
                nn.Linear(config.hidden_size, config.hidden_size),
            )

            setattr(
                self,
                f"layer_norm_{sent_id}",
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

            setattr(self, f"dropout_{sent_id}", nn.Dropout(config.hidden_dropout_prob))

class RobertaMLPDemultiplexerSequenceClassification(nn.Module):
    """Head for sequence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = RobertaMLPDemuxModule(config)
        self.dense_before_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.layernorm_presoftmax = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, features):
        # extract the first <num instance> representations and concatenate with the right word
        _, _, _ = features.shape
        all_feats = []
        for sent_id in range(self.num_instances):
            cur_dense1 = getattr(self.demux_module, f"dense_{sent_id}")
            cur_layer_norm = getattr(self.demux_module, f"layer_norm_{sent_id}")
            dropout = getattr(self.demux_module, f"dropout_{sent_id}")

            cls_feat = features[:, 0, :]
            x = dropout(cls_feat)
            x = cur_dense1(x)
            x = gelu(x)
            x = cur_layer_norm(x)

            all_feats.append(x)

        all_feats = torch.stack(all_feats, dim=1)
        demuxed_representations = all_feats.view(-1, all_feats.shape[-1])
        x = self.dense_before_out_proj(demuxed_representations)
        x = gelu(x)
        x = self.layernorm_presoftmax(x)
        x = self.out_proj(x)

        return x, demuxed_representations

class RetrievalHeadMLPDemultiplexing(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different instances
        self.demux_module = RobertaMLPDemuxModule(config)

        # shared vocab layers across different instances
        self.dense_pre_vocab = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm_pre_vocab = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = nn.Linear(config.hidden_size, math.ceil(config.vocab_size / config.retrieval_loss_vocab_scale))
        self.bias = nn.Parameter(torch.zeros(math.ceil(config.vocab_size / config.retrieval_loss_vocab_scale)))
        self.decoder.bias = self.bias

    def forward(self, features, instance_labels, **kwargs):
        # extract the first <num instance> representations and concatenate with the right word
        batch, seqlength, _ = features.shape
        all_feats = torch.zeros_like(features)
        all_feats = all_feats.view(-1, features.shape[-1])

        for sent_id in range(self.num_instances):
            cur_dense1 = getattr(self.demux_module, f"dense_{sent_id}")
            cur_layer_norm = getattr(self.demux_module, f"layer_norm_{sent_id}")
            dropout = getattr(self.demux_module, f"dropout_{sent_id}")

            cur_sent_mask = instance_labels == sent_id
            cur_sent_feats = features[cur_sent_mask]

            x = dropout(cur_sent_feats)
            x = cur_dense1(x)
            x = gelu(x)
            x = cur_layer_norm(x)

            all_feats[cur_sent_mask.view(-1), :] = x

        # reshape into  B x L x V
        all_feats = all_feats.view(batch, seqlength, -1)
        # project back to size of vocabulary with bias
        x = self.dense_pre_vocab(all_feats)
        x = gelu(x)
        x = self.layer_norm_pre_vocab(x)
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class RobertaMLPDemultiplexerTokenClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        # initialize different MLPs for different sentences
        self.demux_module = RobertaMLPDemuxModule(config)
        # shared vocab layers across different sentences
        self.dense_before_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.layernorm_presoftmax = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, features):
        # extract the first <num sentence> representations and concatenate with the right word
        _, seq_length, feature_dim = features.shape
        all_feats = []
        for sent_id in range(self.num_instances):
            cur_dense1 = getattr(self.demux_module, f"dense_{sent_id}")
            cur_layer_norm = getattr(self.demux_module, f"layer_norm_{sent_id}")
            dropout = getattr(self.demux_module, f"dropout_{sent_id}")
            inp_feat = features
            x = dropout(inp_feat)
            x = cur_dense1(x)
            x = gelu(x)
            x = cur_layer_norm(x)
            all_feats.append(x.unsqueeze(1))

        #  B x L x dim
        # stack to get B x N X L X dim
        all_feats = torch.cat(all_feats, dim=1)
        # collapse the first 2 dimensions
        demuxed_representations = all_feats.view(-1, seq_length, feature_dim)
        
        x = self.dense_before_out_proj(demuxed_representations)
        x = gelu(x)
        x = self.layernorm_presoftmax(x)
        x = self.out_proj(x)

        return x, demuxed_representations

######  DATA CLASSES #######

@dataclass
class SequenceClassifierOutputMuxed(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    task_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    retrieval_predictions: Optional[torch.FloatTensor] = None
    retrieval_instance_labels: Optional[torch.FloatTensor] = None

@dataclass
class TokenClassifierOutputMuxed(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    task_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
