import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, GenerationMixin
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mbart.modeling_mbart import MBartPreTrainedModel, MBartConfig, MBartScaledWordEmbedding, MBartDecoderLayer, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.donut.modeling_donut_swin import DonutSwinModelOutput
from einops import rearrange
from typing import Optional, List, Union, Tuple
import warnings
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from .hf_nemotron_parse_config import NemotronParseConfig
from transformers import AutoModel
import time
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


class NemotronParseDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = MBartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.config = config

        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config._attn_implementation == "sdpa" and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa" and cross_attn_head_mask is None and not output_attentions:
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
        hidden_states = inputs_embeds
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class RadioWithNeck(nn.Module):
    """Vision encoder using RADIO model with custom neck."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model_encoder = AutoModel.from_config(config, trust_remote_code=True)
        
        # Neck components
        last_hidden_state = 1024
        self.conv1 = nn.Conv1d(1280, last_hidden_state, 1)
        self.layer_norm1 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.conv2 = nn.Conv2d(last_hidden_state, last_hidden_state, kernel_size=(1,4), stride=(1,4), padding=0, bias=False)
        self.layer_norm2 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        self.sum_proj = nn.Linear(3840, last_hidden_state)
        self.layer_norm3 = nn.LayerNorm(last_hidden_state, eps=1e-06, elementwise_affine=True)
        
    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=False, **kwargs):
        radio_output = self.model_encoder(pixel_values)
        summary, feature = radio_output
        
        
        output = self.conv1(feature.permute(0,2,1)).permute(0,2,1)
        output = self.layer_norm1(output)
        
        patch_size = self.config.patch_size
        output = rearrange(output, 'b (h w) d -> b d h w',
                    h=pixel_values.shape[-2] // patch_size,
                    w=pixel_values.shape[-1] // patch_size)
        
        output = self.conv2(output)
        output = rearrange(output, 'b d h w -> b (h w) d')
        output = self.layer_norm2(output)
        summary = self.layer_norm3(self.sum_proj(summary))
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)
        
        return DonutSwinModelOutput(last_hidden_state=output)


class NemotronParsePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = NemotronParseConfig
    base_model_prefix = "vision_encoder_decoder"  # Use VisionEncoderDecoder prefix
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RadioWithNeck", "MBartDecoder"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.decoder.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.decoder.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# Based on transformers.models.encoder_decoder.modeling_encoder_decoder
class NemotronParseForConditionalGeneration(NemotronParsePreTrainedModel, GenerationMixin):
    """
    NemotronParse model for conditional generation tasks.
    
    This model combines a RADIO-based vision encoder with an mBART-based text decoder.
    """
    
    def __init__(self, config: NemotronParseConfig):
        super().__init__(config)
        
        self.encoder = RadioWithNeck(config.encoder)
        self.encoder.main_input_name = 'pixel_values'
        self.encoder = self.encoder.to(config.encoder.torch_dtype)
        
        self.decoder = NemotronParseDecoder(config.decoder)
        self.decoder = self.decoder.to(config.decoder.torch_dtype)

        self.lm_head = nn.Linear(config.decoder.d_model, config.decoder.vocab_size, bias=False, dtype=config.decoder.torch_dtype)
 
        # Extra heads
        num_extra_heads = getattr(config, 'num_extra_heads', 0)
        self.decoder.extra_heads = nn.ModuleList([
            nn.Linear(config.decoder.d_model, config.decoder.d_model) 
            for _ in range(num_extra_heads)
        ])
        self.decoder.extra_proj = nn.ModuleList([
            nn.Linear(config.decoder.d_model, config.decoder.d_model)
            for _ in range(num_extra_heads)
        ])
        
        # Class token index for loss weighting
        self.class_token_indx_start = getattr(config, 'class_token_start_idx', 50000)
        
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        __subflavors__: Optional[str] = None,
        __keys__: Optional[List[str]] = None,
        return_sample_losses: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_hidden_states = True

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        loss = None

        if labels is not None:
            main_logits = self.lm_head(decoder_outputs.last_hidden_state)
            logits = [main_logits]
            decoder_inputs_embeds = decoder_outputs.inputs_embeds
            for iii, head in enumerate(self.decoder.extra_heads):

                decoder_input_embeds_shift = self.decoder.extra_proj[iii](torch.cat((decoder_inputs_embeds[:,1:,:], torch.zeros_like(decoder_inputs_embeds[:,0,:].unsqueeze(1))), axis=1))
                hidden = head(decoder_outputs['hidden_states'][-1] + decoder_input_embeds_shift)
                logits.append(self.lm_head(hidden))  # Use main lm_head, NOT decoder.lm_head

            logits = torch.stack(logits, dim=-2)
            loss_fct = CrossEntropyLoss(reduction="none")

            losses_per_head = []
            tokens_per_head = []
            for head_num in range(len(self.decoder.extra_heads)+1):
                logits_head = logits[:,:,head_num,:]
                labels_head = torch.cat(
                    (labels[:, head_num:], torch.full_like(labels[:, :head_num], -100)),
                    1
                )
                loss_full = loss_fct(logits_head.permute(0, 2, 1), labels_head)
                loss_full[labels_head >= self.class_token_indx_start] *= 10
                losses_per_head.append(loss_full.sum(1))
                tokens_per_head.append((labels_head != -100).sum(1))

            losses_per_sample = torch.stack(losses_per_head, dim=1).sum(1)
            tokens_per_sample = torch.stack(tokens_per_head, dim=1).sum(1)
            loss = losses_per_sample.sum() / (tokens_per_sample.sum() + 1e-6)
            if return_sample_losses is not None:
                return_sample_losses.copy_(losses_per_sample.detach() / (tokens_per_sample + 1e-6))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        output_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=output_logits, 
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None):
        """Resize token embeddings and update lm_head accordingly."""
        # Resize decoder embeddings
        new_embeddings = self.decoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        
        # Update lm_head to match new vocab size
        if new_embeddings is not None:
            old_vocab_size, hidden_size = self.lm_head.weight.shape
            new_vocab_size = new_embeddings.num_embeddings
            
            if old_vocab_size != new_vocab_size:
                print(f"Resizing lm_head from {old_vocab_size} to {new_vocab_size} tokens")
                new_lm_head = nn.Linear(hidden_size, new_vocab_size, bias=False, device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)
                
                # Copy old weights to new lm_head
                num_tokens_to_copy = min(old_vocab_size, new_vocab_size)
                new_lm_head.weight.data[:num_tokens_to_copy] = self.lm_head.weight.data[:num_tokens_to_copy]
                
                # Update reference
                self.lm_head = new_lm_head
                # DO NOT update decoder.lm_head - keep them separate
        
        return new_embeddings

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
