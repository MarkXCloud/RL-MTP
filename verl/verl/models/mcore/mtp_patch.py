# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Callable

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    roll_tensor,
)


@dataclass
class MTPRLOutput:
    """Container for per-head MTP log_probs used by RL policy loss."""

    mtp_log_probs: list = field(default_factory=list)

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    from verl.utils.megatron_utils import unwrap_model


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_postprocess(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess_backup = model._postprocess
        model._postprocess = _megatron_gptmodel_postprocess.__get__(model, model.__class__)


def unpatch_postprocess(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess = model._postprocess_backup


def patch_postprocess_rl(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess_backup = model._postprocess
        model._postprocess = _megatron_gptmodel_postprocess_rl.__get__(model, model.__class__)


def unpatch_postprocess_rl(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess = model._postprocess_backup


# copy from https://github.com/NVIDIA/Megatron-LM/blob/23e092f41ec8bc659020e401ddac9576c1cfed7e/megatron/core/models/gpt/gpt_model.py
# patch the postprocess method of GPTModel to support advanced features like MTP, 1f1b overlap, etc.
def _megatron_gptmodel_postprocess(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess and labels is not None:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    # Skip when mtp_num_layers is None or 0
    if self.config.mtp_num_layers and labels is not None:
        mtp_labels = labels.clone()

        hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(mtp_labels)
        for mtp_layer_number in range(self.config.mtp_num_layers):
            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )

            # Compute mtp loss without storing logits to save memory.
            mtp_loss = self.compute_output_layer_and_language_model_loss(
                hidden_states_list[mtp_layer_number + 1],
                labels=mtp_labels,
                weight=self.shared_embedding_or_output_weight(),
                sequence_parallel_enabled=self.output_layer.sequence_parallel,
                column_parallel_linear=self.output_layer,
                col_linear_kwargs={
                    "weight": output_weight,
                    "runtime_gather_output": runtime_gather_output,
                },
            )

            mtp_loss = loss_mask * mtp_loss
            if self.training:
                # TODO(shifangx): remove the use of parallel_state here
                # after moving loss logging to loss_func in pretrain_gpt.py
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    mtp_layer_number,
                    self.config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
            if self.config.calculate_per_token_loss:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
            else:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)

    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    # [s b h] => [b s h]
    return logits.transpose(0, 1).contiguous()


def _megatron_gptmodel_postprocess_rl(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    """RL variant of postprocess: stores per-head MTP log_probs on the model
    instead of computing scaled CE loss via MTPLossAutoScaler.

    Gradients flow naturally when the caller includes the MTP policy loss
    in the total loss that is back-propagated.
    """

    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess and labels is not None:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    if self.config.mtp_num_layers and labels is not None:
        mtp_labels = labels.clone()

        hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]
        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)

        mtp_log_probs = []

        for mtp_layer_number in range(self.config.mtp_num_layers):
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )

            mtp_ce = self.compute_output_layer_and_language_model_loss(
                hidden_states_list[mtp_layer_number + 1],
                labels=mtp_labels,
                weight=self.shared_embedding_or_output_weight(),
                sequence_parallel_enabled=self.output_layer.sequence_parallel,
                column_parallel_linear=self.output_layer,
                col_linear_kwargs={
                    "weight": output_weight,
                    "runtime_gather_output": runtime_gather_output,
                },
            )

            mtp_log_prob = -mtp_ce
            assert mtp_layer_number == 0, "current only support shifts for the first MTP layer"
            aligned_mtp_log_prob, _ = _roll_tensor_packed_seq_right(
                mtp_log_prob,
                shifts=mtp_layer_number + 1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            mtp_log_probs.append(aligned_mtp_log_prob)

        self._mtp_rl_output = MTPRLOutput(mtp_log_probs=mtp_log_probs)

    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    # [s b h] => [b s h]
    return logits.transpose(0, 1).contiguous()


def patch_mtp_layer_get_embeddings(model: torch.nn.Module):
    """Patch the _get_embeddings method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)
    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    if target_layers:
        for layer in target_layers:
            layer._get_embeddings_backup = layer._get_embeddings
            layer._get_embeddings = _patched_get_embeddings_for_detach.__get__(layer, layer.__class__)
        print(f"Found and patched {len(target_layers)} MTP layer(s) in any of the actor modules")
        return True
    else:
        print("No MTP layers found to patch in any of the actor modules")
        return False


def unpatch_mtp_layer_get_embeddings(model: torch.nn.Module):
    """Unpatch the _get_embeddings method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)

    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    unpatched_count = 0
    for layer in target_layers:
        if hasattr(layer, "_get_embeddings_backup"):
            layer._get_embeddings = layer._get_embeddings_backup
            delattr(layer, "_get_embeddings_backup")
            unpatched_count += 1

    if unpatched_count > 0:
        print(f"Unpatched {unpatched_count} MTP layer(s)")
        return True
    return False


def _patched_get_embeddings_for_detach(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    embedding: Callable,
    hidden_states: torch.Tensor,
    packed_seq_params=None,
):
    """
    Patched version of _get_embeddings method for MultiTokenPredictionLayer.

    This is a modified version that you can customize according to your needs.
    The original implementation is preserved below with modifications.
    """

    # You can modify the logic here as needed
    # For example, you could:
    # - Change the shift amount in roll_tensor
    # - Apply custom transformations to input_ids or position_ids
    # - Add debugging information
    # - Modify the embedding computation

    # Original logic with custom modifications
    from megatron.core.transformer.multi_token_prediction import roll_tensor
    from megatron.core.utils import make_viewless_tensor

    # Calc logits for the current Multi-Token Prediction (MTP) layers.
    input_ids, _ = roll_tensor(
        input_ids,
        shifts=-1,  # You can modify this shift value
        dims=-1,
        cp_group=self.cp_group,
        packed_seq_params=packed_seq_params,
    )
    position_ids, _ = roll_tensor(
        position_ids,
        shifts=-1,  # You can modify this shift value
        dims=-1,
        cp_group=self.cp_group,
        packed_seq_params=packed_seq_params,
    )

    # embedding computation - you can modify this part
    decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

    # Apply custom transformations if needed
    # For example: decoder_input = some_custom_function(decoder_input)

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    # detach decoder_input and hidden_states
    decoder_input = decoder_input.detach()
    hidden_states = hidden_states.detach()

    return input_ids, position_ids, decoder_input, hidden_states

def _roll_tensor_packed_seq_right(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed sequence support (right shift by 1).
    This function handles rolling for packed sequences by respecting sequence boundaries.
    This function is modified from the original roll_tensor function in megatron.core.transformer.multi_token_prediction,
    because the original roll_tensor function only supports left shift by 1, and we need to support right shift by 1.
    """

    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."
    assert shifts == 1, "This function only supports a single-token right shift."
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        # CP disabled: roll each packed sequence independently within its boundaries
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            # Zero out the first position(s) that would cross sequence boundaries
            #   torch.roll(shifts=+1): [a0,a1,a2] → [a2,a0,a1]
            #   after zeroing index 0:               [ 0,a0,a1]  ✓
            rolled_seq[..., :shifts] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        return rolled_tensor, rolled_tensor.sum()

    # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size

        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # Split into front chunk (chunk0) and mirrored-tail chunk (chunk1)
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        # ── Prepare boundary tokens ──
        # After torch.roll(shifts=+1), position 0 holds the wrapped-around element.
        # That wrapped element is what we need to send, and position 0 is where we
        # will write the value received from the neighbour.
        boundary_idx = 0  # the "wrong" position after right-shift roll

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            if chunk.size(dims) == 0:
                tensor_send_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                tensor_recv_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                continue
            boundary = chunk.select(dims, boundary_idx).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        # ── P2P communication ──
        # Direction summary for shifts=+1:
        #   chunk0 (front, left→right):  send → next_rank,  recv ← prev_rank
        #   chunk1 (tail,  right←left):  send → prev_rank,  recv ← next_rank
        #
        #       Rank 0          Rank 1          Rank 2          Rank 3
        #     [t0 | t7]      [t1 | t6]      [t2 | t5]      [t3 | t4]
        #
        #  chunk0: zero→t0  t0→t1  t1→t2  t2→t3
        #  chunk1: t7←t6    t6←t5  t5←t4  t4←t3(local chunk0)

        ops = []

        # ---- Communication with prev_rank ----
        if local_rank != 0:
            # chunk0: receive the token that belongs to my position from prev_rank
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=prev_rank))
            # chunk1: send my wrapped boundary to prev_rank
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=prev_rank))
        else:
            # Rank 0: chunk0 is the start of the sequence → fill with zero
            tensor_recv_list[0].zero_()

        # ---- Communication with next_rank ----
        if local_rank != cp_size - 1:
            # chunk0: send my wrapped boundary to next_rank
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=next_rank))
            # chunk1: receive the token that belongs to my position from next_rank
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=next_rank))
        else:
            # Last rank: chunk0 (t3) and chunk1 (t4) are adjacent in the global
            # sequence.  Right-shifting t4 needs t3, which is in local chunk0.
            tensor_recv_list[1].copy_(tensor_send_list[0])

        for op in ops:
            op.wait()

        # ── Write received tokens into the rolled chunks ──
        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = boundary_idx  # position 0
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor, rolled_tensor.sum()
