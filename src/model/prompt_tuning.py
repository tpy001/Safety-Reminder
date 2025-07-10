import torch
import torch.nn as nn
from peft import PromptEmbedding
from loguru import logger
from src.utils import get_class_name
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
import torch
from transformers import LlavaForConditionalGeneration
from typing import Optional, Tuple, Union, List
from .base_model import VQAModel
from .llava import VQALlaVA


    
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
        **kwargs,
    ):

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

        self.word_embedding = self.model.language_model.get_input_embeddings()

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
                self.prompt_embedding.load_state_dict(torch.load(pretrained_path))
                logger.info(f"Loading pretrained prompt embedding from {pretrained_path}...")
            except:
                logger.warning(f"Failed to load pretrained prompt embedding from {pretrained_path}...")

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
            prompt = soft_prompt +" " + prompt
        elif insert_pos == 'last':
            prompt = prompt + " " + soft_prompt
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