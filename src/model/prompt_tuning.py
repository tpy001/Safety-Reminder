from .base_model import BaseModel
import torch
import torch.nn as nn
from peft import PromptEmbedding, PromptTuningConfig,get_peft_model
from loguru import logger
from src.utils import get_class_name,print_trainable_parameters
from transformers import AddedToken
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
import torch
from PIL import Image
from .llava import format_prompt
from transformers.generation import GenerationConfig



    
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

class PromptTuningLlava(BaseModel):
    '''
        Add the soft prompt to the beginning of the user question.
        The loss function: Auto-regression language model loss
    ''' 
    def __init__(self, peft_config=None,vlm_model=None,pretrained_path=None,add_pos="first",*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = vlm_model
        self.add_pos = add_pos
        
        # modify the default forward function of llava in order to support prompt tuning
        self.alter_forward_behavior()
       
        self.soft_prompt_num =  peft_config['num']
        self.soft_prompt_dim =  peft_config['dim']
        self.soft_prompt_text =  peft_config['prompt_text']
        self.soft_prompt_init_text =  peft_config['init_text']
        if self.soft_prompt_text not in self.model.tokenizer.get_vocab():
            self.model.tokenizer.add_tokens([self.soft_prompt_text])
        self.soft_prompt_id = self.model.tokenizer.convert_tokens_to_ids(self.soft_prompt_text)

        self.word_embedding = self.model.get_language_model().get_input_embeddings()
        # Step 1: tokenize init text
        init_tokens = self.model.tokenizer(self.soft_prompt_init_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

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

        # add new attributes to the base model for prompt tuning
        self.add_new_attr(
            dict(
                soft_prompt_id = self.soft_prompt_id,
                prompt_embedding = self.prompt_embedding,
                soft_prompt_num =  self.soft_prompt_num,
            )
        )

        self.load_pretrained(pretrained_path)
        print_trainable_parameters(self)

    def load_pretrained(self,pretrained_path=None):        
        if pretrained_path is not None:
            try:
                self.prompt_embedding.load_state_dict(torch.load(pretrained_path))
                logger.info(f"Loading pretrained prompt embedding from {pretrained_path}...")
            except:
                logger.warning(f"Failed to load pretrained prompt embedding from {pretrained_path}...")


    
    def alter_forward_behavior(self):
        '''
            modify the default forward function of llava in order to support prompt tuning
        '''
        self.model.model.forward= llava_forward_prompt_tuning.__get__( self.model.model,type(self.model.model))
        logger.info(f"Modifying the forward function of {get_class_name(self.model.model)} to 'llava_forward_prompt_tuning' in order to support prompt tuning...")
    
    def add_new_attr(self,dict):
        '''
            add new attributes to the base model for prompt tuning
        '''
        for name, value in dict.items():
            setattr(self.model.model, name, value)
            logger.info(f"Adding attributes {name}={value} to {get_class_name(self.model.model)} for prompt tuning...")

    def freeze_base_model(self):
        # freeze the basic model and only fine-tuning the learnable prompt
        logger.info(f"Preparing for prompt tuning...")
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"Freezing the {get_class_name(self.model)} model and only fine-tuning the learnable prompt...")

    def train(self,mode = True):
        super().train(False)


    def add_soft_prompt(self,prompt):
        insert_pos = self.add_pos
        soft_prompt = " ".join([self.soft_prompt_text]*self.soft_prompt_num)
        if insert_pos == 'first':
            prompt = soft_prompt +" " + prompt
        elif insert_pos == 'last':
            prompt = prompt + " " + soft_prompt
        else:
            raise ValueError("insert_pos must be 'first' or 'last'")
        return prompt
    
    def forward(self, inputs,use_image=True):
        batch_prompts, images = self.model.get_formated_prompt(inputs, use_image)
        batch_prompts = [ self.add_soft_prompt(prompt) for prompt in batch_prompts]
        return self.model._forward(batch_prompts, images=images)

    def generate(self,inputs, use_image = True):
        batch_prompts, images = self.model.get_formated_prompt(inputs, use_image)
        batch_prompts = [ self.add_soft_prompt(prompt) for prompt in batch_prompts]
        return self.model.generate_text(batch_prompts, images=images)
    
    def save_model(self,path):
        torch.save(self.prompt_embedding.state_dict(), path)

    def parameters(self, recurse: bool = True):
        return self.prompt_embedding.parameters()

from typing import Optional, Tuple, Union, List


def llava_forward_prompt_tuning(
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
):

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

    legacy_processing = False
    
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
        # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
        # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
        legacy_processing = (
            (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
        ) or (input_ids.shape[-1] == 1 and pixel_values is not None)


        soft_prompt_mask = (
            (input_ids == self.soft_prompt_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        index = torch.arange(self.soft_prompt_num)
        soft_prompt_embeddings = self.prompt_embedding(index).expand(inputs_embeds.shape[0], self.soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,soft_prompt_embeddings )

    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        if legacy_processing:
            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
            # prefill stage vs decoding stage (legacy behavior copied)
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[
                    -target_length:
                ]

        # TODO: @raushan retain only the new behavior after v4.47
        else:
            n_image_tokens = (input_ids == self.config.image_token_index).sum(dim=-1)[0].item()
            n_image_features = image_features.shape[1]
            if n_image_tokens != n_image_features:
                # logger.warning(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
                # image_features = image_features[:,-n_image_tokens:]
                # n_image_features = n_image_tokens
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            n_soft_prompt = (input_ids == self.soft_prompt_id).sum(dim=-1)[0].item()
            if n_soft_prompt != self.soft_prompt_num and n_soft_prompt > 0:
                raise ValueError(
                    f"Soft prompt tokens do not match: tokens: {n_soft_prompt}, expected: {self.soft_prompt_num}"
                )
            
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            soft_prompt_mask = (
                (input_ids == self.soft_prompt_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            # soft prompt token embeddings
            index = torch.arange(self.soft_prompt_num)
            soft_prompt_embeddings = self.prompt_embedding(index).expand(inputs_embeds.shape[0], self.soft_prompt_num, -1).to(inputs_embeds.device,inputs_embeds.dtype)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(soft_prompt_mask,soft_prompt_embeddings )

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


        