import torch
import torch.nn as nn
from peft import PromptEmbedding
from loguru import logger
from src.utils import get_class_name
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch
from transformers import LlavaForConditionalGeneration,Qwen2VLForConditionalGeneration
from typing import Optional, Tuple, Union, List
from .base_model import VQAModel
from .llava import VQALlaVA
from .Qwen2VL import VQAQwen2VL
from transformers.generation import GenerationConfig
from transformers.cache_utils import *

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList  
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
    
class PromptEmbedding(torch.nn.Module):
    def __init__(self, num, dim, init_embeds=None):
        super().__init__()
        self.soft_prompt_num = num
        self.embedding_dim = dim
        self.embedding = torch.nn.Embedding(num, dim,dtype=torch.float32)

        if init_embeds is not None:
            if init_embeds.shape != (num, dim):
                raise ValueError(f"init_embeds shape {init_embeds.shape} does not match ({num}, {dim})")
            self.embedding.weight.data = init_embeds.clone()

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices.to(self.embedding.weight.device))
        return prompt_embeddings


class PromptTuningLlavaModel(LlavaForConditionalGeneration):
     def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        soft_prompt_id = None,
        soft_prompt_num = None,
        soft_prompt_embedding = None,
        use_original_sample = False, # Unused, kept for compatibility with transformers library
        use_original_forward = False,
        **kwargs,
    ):
        if use_original_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )
        
        assert soft_prompt_id is not None
        assert soft_prompt_num is not None
        assert soft_prompt_embedding is not None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True

            soft_prompt_mask = (
                (input_ids == soft_prompt_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            index = torch.arange(soft_prompt_num)
            _soft_prompt_embeddings = soft_prompt_embedding(index).expand(inputs_embeds.shape[0], soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,_soft_prompt_embeddings )

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            n_image_tokens = (input_ids == self.config.image_token_index).sum(dim=-1)[0].item()
            n_image_features = image_features.shape[1]
            if n_image_tokens != n_image_features:
                # logger.warning(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
                # image_features = image_features[:,-n_image_tokens:]
                # n_image_features = n_image_tokens
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            n_soft_prompt = (input_ids == soft_prompt_id).sum(dim=-1)[0].item()
            if n_soft_prompt != soft_prompt_num and n_soft_prompt > 0:
                raise ValueError(
                    f"Soft prompt tokens do not match: tokens: {n_soft_prompt}, expected: {soft_prompt_num}"
                )
            
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            soft_prompt_mask = (
                (input_ids == soft_prompt_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            # soft prompt token embeddings
            index = torch.arange(soft_prompt_num)
            _soft_prompt_embeddings = soft_prompt_embedding(index).expand(inputs_embeds.shape[0], soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,_soft_prompt_embeddings )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class PromptTuningQwen2VLModel(Qwen2VLForConditionalGeneration):
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
        use_original_forward = False,
        **kwargs,
    ):  
        if use_original_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                **kwargs,
            )
        
        assert soft_prompt_id is not None
        assert soft_prompt_num is not None
        assert soft_prompt_embedding is not None
    
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


class PromptTuning(VQAModel):
    '''
        Add the soft prompt to the beginning of the user question.
        The loss function: Auto-regression language model loss
    ''' 
    def __init__(self, peft_config=None,pretrained_path=None,add_pos="first",*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.add_pos = add_pos
        
        self.soft_prompt_num =  peft_config['num']
        self.soft_prompt_dim =  peft_config['dim']
        self.soft_prompt_init_text =  peft_config['init_text']
        self.soft_prompt_text = peft_config['prompt_text']
        if self.soft_prompt_text not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.soft_prompt_text])
        self.soft_prompt_id = self.tokenizer.convert_tokens_to_ids(self.soft_prompt_text)

        self.word_embedding = self.get_language_model().get_input_embeddings()

        # Step 1: tokenize init text
        init_tokens = self.tokenizer(self.soft_prompt_init_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        # Step 2: get corresponding word embeddings
        init_embeds = self.word_embedding(init_tokens.to(self.word_embedding.weight.device)).float()  # shape: [L, D]

        # Step 3: repeat or truncate to match desired soft_prompt_num
        if len(init_embeds) >= self.soft_prompt_num:
            init_embeds = init_embeds[:self.soft_prompt_num]
        else:
            repeat_times = (self.soft_prompt_num + len(init_embeds) - 1) // len(init_embeds)
            init_embeds = init_embeds.repeat((repeat_times, 1))[:self.soft_prompt_num]

        # Step 4: initialize prompt embedding module
        self.prompt_embedding = PromptEmbedding(self.soft_prompt_num, self.soft_prompt_dim, init_embeds)

        # freeze the basic model and only fine-tuning the learnable prompt
        self.freeze_base_model()

        self.load_pretrained(pretrained_path)


    def load_pretrained(self,pretrained_path=None):        
        if pretrained_path is not None:
            try:
                self.prompt_embedding.load_state_dict(torch.load(pretrained_path,weights_only=True))
                logger.info(f"Loading pretrained prompt embedding from {pretrained_path}...")
            except:
                logger.warning(f"Failed to load pretrained prompt embedding from {pretrained_path}...")
                raise

    def freeze_base_model(self):
        # freeze the basic model and only fine-tuning the learnable prompt
        logger.info(f"Preparing for prompt tuning...")
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"Freezing the {get_class_name(self.model)} model and only fine-tuning the learnable prompt...")


    def add_soft_prompt(self,prompt,add_pos="first"):
        insert_pos = add_pos
        soft_prompt = " ".join([self.soft_prompt_text]*self.soft_prompt_num)
        if insert_pos == 'first':
            prompt = soft_prompt.rstrip() +" " + prompt.lstrip()
        elif insert_pos == 'last':
            prompt = prompt.rstrip() + " " + soft_prompt.lstrip()
        else:
            raise ValueError("insert_pos must be 'first' or 'last'")
        return prompt
    
    def forward(self,inputs,use_image=True,output_hidden_states=False):
        self.check_inputs(inputs,use_image)
        batch_prompts,images = self.get_formatted_prompt(inputs,use_image)
        formatted_prompt = [ self.add_soft_prompt(prompt) for prompt in batch_prompts]
        return self._forward(
            formatted_prompt,
            images=images,
            output_hidden_states=output_hidden_states,
            soft_prompt_id = self.soft_prompt_id,
            soft_prompt_num = self.soft_prompt_num,
            soft_prompt_embedding = self.prompt_embedding
        )

    def generate(self,inputs, use_image = True,return_full_text=False,**kwargs):
        batch_prompts, images = self.get_formatted_prompt(inputs, use_image)
        batch_prompts = [ self.add_soft_prompt(prompt,add_pos="first") for prompt in batch_prompts]
        return self.generate_text(
            batch_prompts, 
            images=images,
            return_full_text=return_full_text,
            soft_prompt_id = self.soft_prompt_id,
            soft_prompt_num = self.soft_prompt_num,
            soft_prompt_embedding = self.prompt_embedding,
            **kwargs
        )
    
    
    def save_model(self,path):
        torch.save(self.prompt_embedding.state_dict(), path)

    def parameters(self, recurse: bool = True):
        return self.prompt_embedding.parameters()

   
class PromptTuningLlava(PromptTuning,VQALlaVA):
    model_cls = PromptTuningLlavaModel

class PromptTuningQwen2VL(PromptTuning,VQAQwen2VL):
    model_cls = PromptTuningQwen2VLModel
