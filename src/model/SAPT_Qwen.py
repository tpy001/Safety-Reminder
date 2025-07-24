from .SAPT import check_if_tokens_start_with_capital,get_prefix
from .prompt_tuning import PromptTuning
from transformers import Qwen2VLForConditionalGeneration
import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList  
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers import AutoProcessor
from transformers.cache_utils import *
import  torch.nn as nn
from PIL import Image
from .Qwen2VL import format_prompt
from transformers.generation import GenerationConfig
from copy import deepcopy
import numpy as np

def classification(hidden_states: torch.Tensor, classfier_data: dict, top_m = 4) -> torch.Tensor:
    device = hidden_states.device  
    
    pca_components = torch.tensor(classfier_data["pca_components"][:top_m], dtype=torch.float32, device=device)  # (m, D)
    pca_mean = torch.tensor(classfier_data["pca_mean"], dtype=torch.float32, device=device)                       # (D,)
    clf_weights = torch.tensor(classfier_data["clf_weights"], dtype=torch.float32, device=device)                 # (1, m)
    clf_bias = torch.tensor(classfier_data["clf_bias"], dtype=torch.float32, device=device)                       # (1,)

    centered_features = hidden_states - pca_mean           # shape: (N, D)
    projected = centered_features @ pca_components.T       # shape: (N, m)
    logits = projected @ clf_weights.T + clf_bias          # shape: (N, 1)
    probs_pos = torch.sigmoid(logits)                      # shape: (N, 1)

    return probs_pos


class SAPTQwen2VLModel(Qwen2VLForConditionalGeneration):
    def validate_is_safe(self,input_ids,model_kwargs,logits_processor):
        """
            Validate if the current generations are safe or not using our trained safety prompt
        """
        soft_prompt_id = model_kwargs["soft_prompt_id"]
        soft_prompt_num = model_kwargs["soft_prompt_num"]
        batch_size = input_ids.shape[0]
        new_model_kwargs = deepcopy(model_kwargs)
        if "past_key_values" in new_model_kwargs.keys():
            new_model_kwargs.pop("past_key_values")
        # model_kwargs.pop("cache_position")
        new_model_kwargs["use_cache"] = False
        new_model_inputs = self.prepare_inputs_for_generation(input_ids, **new_model_kwargs)
        new_model_inputs["pixel_values"] = model_kwargs["pixel_values"]

        
        soft_prompt_ids = torch.full(
            (input_ids.shape[0], soft_prompt_num),  # shape: [batch_size, soft_prompt_num]
            fill_value=soft_prompt_id,
            dtype=input_ids.dtype,
            device=input_ids.device  
        )

        # modify the input_ids 
        new_model_inputs["input_ids"] = torch.cat([input_ids,soft_prompt_ids], dim=1)

        # modify the position_ids to include the soft prompt tokens
        position_ids = new_model_inputs["position_ids"] # 3 x batchsize x current length
        last_pos = position_ids[0, :, -1]
        soft_range = torch.arange(soft_prompt_num, device=position_ids.device) 
        soft_position_ids = (last_pos.unsqueeze(1) + 1) + soft_range.unsqueeze(0) 
        soft_position_ids = soft_position_ids.expand(3, -1, -1)
        full_position_ids = torch.cat([position_ids, soft_position_ids], dim=-1)  # # 3 x batchsize x ( current length + soft prompt length
        new_model_inputs["position_ids"] = full_position_ids
    
        # modify the attention_mask
        attention_mask = new_model_inputs["attention_mask"]

        soft_attention_mask = torch.ones(
            (batch_size, soft_prompt_num),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([attention_mask, soft_attention_mask], dim=1)  # [B, T+S]
        new_model_inputs["attention_mask"] = full_attention_mask

        # disable use_cache to avoid unexpected behavior
        new_model_inputs["use_cache"] = False
        new_model_inputs.pop("past_key_values")

        # new_model_inputs.pop("cache_position")
        with torch.no_grad():
            outputs = self(**new_model_inputs, return_dict=True,output_hidden_states=True)

        selected_hidden = outputs.hidden_states[-1][:,-1]
        pred_safe = classification(selected_hidden,self.classfier).squeeze()
        threshold = 0.2
        is_safe = pred_safe >= threshold 
        return is_safe
    
        # next_token_logits = outputs.logits.clone()[:, -1, :].float()
        # next_token_logits = next_token_logits.to(input_ids.device)

        # # pre-process distribution
        # next_token_scores = logits_processor(input_ids, next_token_logits)

        # next_tokens = torch.argmax(next_token_scores, dim=-1)

        # input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # END_CHARS = {"!", "?", ".", "\n", "…", ":"}
        # last_chars = [ text[-1] for text in input_text]
        # is_ended = torch.tensor(    [char in END_CHARS for char in last_chars]).to(input_ids.device)
        # # next_token_strings = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=False)
        # is_capital_list = check_if_tokens_start_with_capital(next_tokens, self.tokenizer)

        # is_safe = (~is_capital_list) | is_ended

        # return is_safe

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        use_original_sample = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        if use_original_sample:
            return Qwen2VLForConditionalGeneration._sample(self,input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, **model_kwargs)
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        
        count = 0
        interval = 16
        is_safe = torch.tensor( [True] * (batch_size)).to(input_ids.device)
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
           
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if count % interval == (interval-1):
            # if count % interval == 0:
                safe = self.validate_is_safe(input_ids,model_kwargs,logits_processor)
                is_safe = is_safe & safe

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)

            # conside the safety of the generated tokens. If unsafe, stop the generation loop.
            unfinished_sequences = unfinished_sequences & is_safe

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            count += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        self.is_safe = is_safe
        print(is_safe.to("cpu"))
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
                **kwargs  # 合并进这个字典
            }
        )
        return model_inputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        soft_prompt_id = None,
        soft_prompt_num = None,
        soft_prompt_embedding = None,
        use_original_sample = False, # Unused, kept for compatibility with transformers library
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            index = torch.arange(soft_prompt_num)
            soft_prompt_embeddings = soft_prompt_embedding(index).expand(inputs_embeds.shape[0], soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)
            soft_prompt_mask = (
                    (input_ids == soft_prompt_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,soft_prompt_embeddings )

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                
                n_soft_prompt = (input_ids == soft_prompt_id).sum(dim=-1)[0].item()
                if n_soft_prompt != soft_prompt_num and n_soft_prompt > 0:
                    raise ValueError(
                        f"Soft prompt tokens do not match: tokens: {n_soft_prompt}, expected: {soft_prompt_num}"
                )
                soft_prompt_mask = (
                    (input_ids == soft_prompt_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )

                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                index = torch.arange(soft_prompt_num)
                soft_prompt_embeddings = soft_prompt_embedding(index).expand(inputs_embeds.shape[0], soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,soft_prompt_embeddings )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        else:
            index = torch.arange(soft_prompt_num)
            soft_prompt_embeddings = soft_prompt_embedding(index).expand(inputs_embeds.shape[0], soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)
            soft_prompt_mask = (
                    (input_ids == soft_prompt_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,soft_prompt_embeddings )
            
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )


class SAPTQwen2VL(PromptTuning):
    model_cls = SAPTQwen2VLModel
    assistant_tag = 'assistant\n'
    classfier_path = "assets/Qwen/pca_classifier.npz"
    classfier_data = np.load(classfier_path)

    add_eos_token = False
    def __init__(self, min_pixels = None,max_pixels = None,*args, **kwargs):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        super().__init__(*args, **kwargs)
        self.model.tokenizer = self.tokenizer 
        print('Load Model Successfully.')

    def get_language_model(self):
        return self.model.model
    

    def load_model(self, *args, **kwargs):
        if self.device not in ['cpu', 'cuda', 'auto', None]:
            raise ValueError("device should be 'cpu', 'cuda', 'auto', or None'.")

        if self.device is None or self.device == 'auto':
            model = self.model_cls.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, device_map='auto',*args, **kwargs
            )
            self.device = model.device
        else:
            model = self.model_cls.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, device_map=self.device,*args, **kwargs
            )

        # This line is modified to add min_pixels and max_pixels to the processor
        processor = AutoProcessor.from_pretrained(self.model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        # Modified end.

        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        self.processor = processor
        self.tokenizer = processor.tokenizer

        if self.tokenizer.chat_template is None:
            assert self.chat_template is not None, "chat_template is not provided in tokenizer."
            self.tokenizer.chat_template = self.chat_template
        return model
 

    def train(self,mode=True):
        if self.trainable is None:
            raise ValueError("trainable should be set to 'all', 'visual_encoder', or 'language_model' in config file.")
        elif self.trainable == 'all':
            self.model.train()
        elif self.trainable == 'visual_encoder':
            self.model.visual_encoder.train()
        elif self.trainable == 'language_model':
            self.model.language_model.train()
        elif self.trainable == 'frozen':
            self.model.eval()

    def get_labels(self, prompts, return_tensor=True,**kwargs):
        """
        Create labels tensor for loss calculation, masking tokens before the last occurrence 
        of the soft prompt token with -100.

        Args:
            prompts (List[Tensor or List[int]]): Batch of tokenized prompts (list of ids or tensors).
            return_tensor (bool): Whether to return a torch.Tensor or list.

        Returns:
            torch.Tensor or List[List[int]]: Masked labels for training.
        """
        # Encode soft prompt text to token id
        target_token =  self.soft_prompt_id

        labels = []

        for prompt in prompts:
            # Convert prompt to list if tensor
            prompt_ids = prompt.tolist() if hasattr(prompt, "tolist") else prompt

            # Find last occurrence of target_token
            position = -1
            for i in range(len(prompt_ids) - 1, -1, -1):
                if prompt_ids[i] == target_token:
                    position = i
                    break
            if position == -1:
                # If token not found, mask all tokens
                raise ValueError(f"Soft prompt token {target_token} not found in prompt")

            # Mask tokens before position with -100
            label = [-100] * (position+1) + prompt_ids[position+1:]
            labels.append(label)

        if return_tensor:
            return torch.tensor(labels)
        else:
            return labels

    def forward(self, inputs,use_image=True,output_hidden_states=False):
        if output_hidden_states:
            raise NotImplementedError("output_hidden_states is not implemented for SAPT-Llava")
            batch_prompts_chosen, images = self.get_formatted_prompt(inputs, use_image)
        else:
            batch_prompts_chosen, batch_prompts_rejected, images = self.get_formatted_prompt_train(inputs, use_image,add_soft_prompt=True)
        try:
            chosen_loss,hidden_states = self._forward(
                batch_prompts_chosen.copy(), 
                images=images,
                # output_hidden_states=output_hidden_states,
                output_hidden_states=True,
                soft_prompt_id = self.soft_prompt_id,
                soft_prompt_num = self.soft_prompt_num,
                soft_prompt_embedding = self.prompt_embedding,)
            
            hidden_states = hidden_states[-1] # Last Layer, 
            # Calculation Classifier Loss
            cls_loss = self.cal_cls_loss(
                batch_prompts_chosen,
                images = images,
                hidden_states = hidden_states,
                labels = inputs["safe"]
            )
            loss = {
                "cls_loss": cls_loss,
                "lm_loss": chosen_loss,
                "total": 0.05*cls_loss + chosen_loss
            }

        except:
            logger.info("Error in forward function of SAPT")
            loss = torch.tensor(0.0, device=getattr(self, 'device', 'cpu'), requires_grad=True).clone()
        if output_hidden_states:
            return loss, hidden_states
        else:
            return loss
    
    def cal_cls_loss(
        self,
        batch_prompts_chosen,
        images,
        hidden_states,
        labels,
    ):
        inputs = self.processor(images=images, text=batch_prompts_chosen, padding=True, return_tensors="pt").to(self.device)
        label_ids = self.get_labels(inputs['input_ids'],substring = self.assistant_tag).to(self.device)
        mask = (label_ids != -100) 
        first_valid_indices = mask.float().masked_fill(~mask, float('inf')).argmin(dim=1)

        token_index = first_valid_indices - 1
        valid_mask = token_index < self.max_context_length

        # Apply valid_mask
        hidden_states = hidden_states[valid_mask]
        token_index = token_index[valid_mask]
        labels = labels.to(valid_mask)[valid_mask]

        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        selected_hidden = hidden_states[batch_indices, token_index]

        loss_fn = nn.BCELoss()
        logits = classification(selected_hidden,self.classfier_data).squeeze(-1)  # shape: (B,)
        labels = labels.float()  # 转为 float tensor
        loss = loss_fn(logits, labels)
        return loss


    def generate(self,inputs, use_image = True,return_full_text = False, **kwargs): 
        batch_prompts, images = self.get_formatted_prompt(inputs, use_image)

        self.model.classfier = self.classfier_data
        first_generated_text = self.generate_text(
            batch_prompts, 
            images = images, 
            return_full_text = return_full_text,
            soft_prompt_id = self.soft_prompt_id,
            soft_prompt_num = self.soft_prompt_num,
            soft_prompt_embedding = self.prompt_embedding,
            **kwargs
        )
        
        is_safe = self.model.is_safe
        inputs["chosen"] = []
        if isinstance(inputs["question"],str):
            inputs["question"] = [inputs["question"]]
        if isinstance(inputs["image"],str) or isinstance(inputs["image"],Image.Image):
            inputs["image"] = [inputs["image"]]
        for i in range(len(is_safe)):
            if not is_safe[i]:
                inputs["chosen"].append(self.add_soft_prompt(first_generated_text[i],add_pos="last"))
            else:
                inputs["chosen"].append(None)

        filtered_inputs = {k: [] for k in inputs}
        for i in range(len(is_safe)):
            if inputs["chosen"][i] is not None:
                for k in inputs:
                    filtered_inputs[k].append(inputs[k][i])

        if len(filtered_inputs["question"]) > 0:
            batch_prompts, images = self.get_formatted_prompt(filtered_inputs, use_image)

            second_generated_text = self.generate_text(
                batch_prompts, 
                images = images, 
                return_full_text = return_full_text,
                soft_prompt_id = self.soft_prompt_id,
                soft_prompt_num = self.soft_prompt_num,
                soft_prompt_embedding = self.prompt_embedding,
                use_original_sample = True,
                **kwargs
            )
            final_outputs = []
            unsafe_idx = 0

            for i in range(len(is_safe)):
                if not is_safe[i]:
                    final_outputs.append(second_generated_text[unsafe_idx])
                    unsafe_idx += 1
                else:
                    final_outputs.append(first_generated_text[i])
            return final_outputs
        else:
            return first_generated_text
    
    def get_formatted_prompt(self,inputs, use_image = True):
        questions = inputs['question']
        if "chosen" in inputs.keys():
            answers = inputs['chosen']
        else:
            answers = None
        if not isinstance(questions,list):
            questions = [questions]
        if answers is not None and not isinstance(answers,list):
            answers = [answers]
        if use_image:
            assert 'image' in inputs.keys(), "Image is not provided in inputs."
            images = inputs['image'] 
            if isinstance(images,str) or isinstance(images,Image.Image) :
                images = [images]
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]

        else:
            images = None

        if answers is None:
            batch_prompts = format_prompt(self.tokenizer,questions,answers = None ,add_generation_prompt=True,use_image=use_image)
        elif self.add_eos_token:
            batch_prompts = format_prompt(self.tokenizer,questions,answers = answers,add_generation_prompt=False,use_image=use_image)
        elif not self.add_eos_token:
            batch_prompts = format_prompt(self.tokenizer,questions,answers = None ,add_generation_prompt=True,use_image=use_image)
            batch_prompts = [ batch_prompts[i]  + answers[i].strip() for i in range(len(batch_prompts))]
        return batch_prompts,images
    
    def add_soft_prompt(self,prompt,add_pos="first"):
        insert_pos = add_pos
        soft_prompt = self.soft_prompt_text * self.soft_prompt_num
        if insert_pos == 'first':
            prompt = soft_prompt  + prompt
        elif insert_pos == 'last':
            prompt = prompt  + soft_prompt
        else:
            raise ValueError("insert_pos must be 'first' or 'last'")
        return prompt
    
    def get_formatted_prompt_train(
        self,
        inputs: dict,
        use_image: bool = True,
        add_soft_prompt: bool = True,
    ):
        """
        Prepare formatted chosen and rejected inputs from a dictionary.

        Args:
            inputs (dict): Should contain the following keys:
                - 'question': List[str]
                - 'label': List[str] with values like "safe" or "harmful"
                - 'chosen': List[str]
                - 'rejected': List[str]
                - 'image': Optional[List[Any]]
            use_image (bool): Whether to use image input in formatting.
            add_soft_prompt (bool): Whether to append soft prompts.

        Returns:
            Tuple[List[str], List[str], Optional[List[Any]]]:
                - formatted_chosen: list of full input strings with preferred answers
                - formatted_rejected: list of full input strings with non-preferred answers
                - images: image input list duplicated if needed, or None
        """
        questions = inputs["question"]
        safe_labels = inputs["safe"]
        chosen_answers = inputs["chosen"]
        rejected_answers = inputs["rejected"]
        images = inputs.get("image", None)

        formatted_chosen = []
        formatted_rejected = []
        
        
        for i in range(len(questions)):
            # Determine which is the preferred answer
            if safe_labels[i]:
                prefix,suffix = get_prefix(chosen_answers[i], return_suffix=True)
            else:
                prefix,suffix = get_prefix(rejected_answers[i], return_suffix=True)

            prompt_inputs = {
                "question": inputs["question"][i],
                "chosen": prefix,
                "image": inputs["image"][i],
            }
           
            formatted_prompt, _ = self.get_formatted_prompt(prompt_inputs, use_image)
            formatted_prompt = formatted_prompt[0]
            if add_soft_prompt:
                formatted_prompt = self.add_soft_prompt(formatted_prompt, add_pos="last") 

            if safe_labels[i]:
                formatted_chosen.append(f"{formatted_prompt.strip()} {suffix.strip()}" + "<|im_end|>\n")
                formatted_rejected.append(f"{formatted_prompt.strip()} {rejected_answers[i].strip()}" + "<|im_end|>\n")
            else:
                formatted_chosen.append(f"{formatted_prompt.strip()} {chosen_answers[i].strip()}" + "<|im_end|>\n")
                formatted_rejected.append(f"{formatted_prompt.strip()} {suffix.strip()}" + "<|im_end|>\n")
           
        if use_image:
            assert 'image' in inputs.keys(), "Image is not provided in inputs."
            images = inputs['image'] 
            if isinstance(images,str) or isinstance(images,Image.Image) :
                images = [images]
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]
        return formatted_chosen, formatted_rejected, images

