import torch
import torch.nn as nn
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration
from transformers.generation import GenerationConfig
import random
from typing import Union, Tuple,List,Optional,Any
from copy import deepcopy
from .prompt_tuning import PromptTuning,PromptTuningLlavaModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList  
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
from transformers.cache_utils import *
from .llava import VQALlaVA,format_prompt,LlaVA_chat_template
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast


def check_if_tokens_start_with_capital(
    token_ids,
    tokenizer
) -> torch.Tensor:
    """
    Checks a batch of token IDs to determine if their corresponding text
    representations start with a capital letter.

    Args:
        token_ids (torch.Tensor): A 1D tensor of token IDs, with shape [batch_size].
        tokenizer (PreTrainedTokenizer): The tokenizer used to decode the token IDs.

    Returns:
        torch.Tensor: A boolean tensor of shape [batch_size]. True if the
                      corresponding token text starts with a capital letter,
                      False otherwise.
    """
    if token_ids.dim() != 1:
        raise ValueError("Input tensor token_ids must be 1-dimensional.")

    # 1. Convert the list of token IDs to their respective text representations.
    #    e.g., [65, 897, 345] -> ['A', ' good', ' day']
    token_strings = tokenizer.batch_decode(token_ids)

    # 2. Iterate through each token string to check if its first letter is uppercase.
    is_capital_list = []
    for token_str in token_strings:
        # Important: Many tokenizers add a special prefix (like ' ' or 'Ġ') to
        # represent a space before the word. We must strip these prefixes
        # before checking the first character.
        # lstrip() will remove all specified leading characters.
        cleaned_str = token_str.lstrip(' ')

        if cleaned_str and cleaned_str[0].isupper():
            is_capital_list.append(True)
        else:
            is_capital_list.append(False)

    # 3. Convert the list of booleans back to a PyTorch tensor and move it
    #    to the same device as the input tensor.
    return torch.tensor(is_capital_list, device=token_ids.device)

def get_prefix(
    answer: str,
    empty_prefix_prob: float = 1/3,
    max_attempts: int = 10,
    min_length: int = 20,
    return_suffix: bool = False
) -> Union[str, Tuple[str, str]]:
    """
    Randomly extract a prefix from `answer` that ends at a space,
    with a probability to return an empty string.
    
    Optionally return the remaining suffix.

    Args:
        answer (str): The full input string.
        empty_prefix_prob (float): Probability of returning an empty prefix.
        max_attempts (int): Max attempts to find a space break point.
        return_suffix (bool): Whether to return the remaining suffix.

    Returns:
        str or (str, str): Prefix only, or (prefix, suffix) if return_suffix is True.
    """
    if random.random() < empty_prefix_prob:
        return ("", answer.strip()) if return_suffix else ""

    for _ in range(max_attempts):
        if len(answer) < min_length:
            length = len(answer)  # fallback to full length
        else:
            length = random.randint(min_length, len(answer))
        prefix_candidate = answer[:length]
        last_space = prefix_candidate.rfind(' ')
        if last_space != -1:
            prefix = prefix_candidate[:last_space].strip()
            suffix = answer[last_space:].strip()
            return (prefix, suffix) if return_suffix else prefix

    # If no space is found
    return ("", answer.strip()) if return_suffix else ""

class SAPTLlavaModel(LlavaForConditionalGeneration):
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
        position_ids = new_model_inputs["position_ids"]
        last_pos = position_ids[:, -1]
        soft_range = torch.arange(soft_prompt_num, device=position_ids.device) 
        soft_position_ids = (last_pos.unsqueeze(1) + 1) + soft_range.unsqueeze(0) 
        full_position_ids = torch.cat([position_ids, soft_position_ids], dim=1)  # [B, T+S]
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
        # new_model_inputs.pop("past_key_values")

        new_model_inputs.pop("cache_position")
        with torch.no_grad():
            outputs = self(**new_model_inputs, return_dict=True)

        next_token_logits = outputs.logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        next_tokens = torch.argmax(next_token_scores, dim=-1)

        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        END_CHARS = {"!", "?", ".", "\n", "…", ":"}
        last_chars = [ text[-1] for text in input_text]
        is_ended = torch.tensor(    [char in END_CHARS for char in last_chars]).to(input_ids.device)
        # next_token_strings = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
        is_capital_list = check_if_tokens_start_with_capital(next_tokens, self.tokenizer)

        is_safe = (~is_capital_list) | is_ended

        return is_safe

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
            return LlavaForConditionalGeneration._sample(self,input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, **model_kwargs)
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

class SAPTLlava(PromptTuning):
    model_cls = SAPTLlavaModel
    assistant_tag = "ASSISTANT:"
    chat_template = LlaVA_chat_template
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.tokenizer = self.tokenizer

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

        batch_prompts_chosen, batch_prompts_rejected, images = self.get_formatted_prompt_train(inputs, use_image,add_soft_prompt=True)
        try:
            chosen_loss = self._forward(
                batch_prompts_chosen, 
                images=images,
                output_hidden_states=output_hidden_states,
                soft_prompt_id = self.soft_prompt_id,
                soft_prompt_num = self.soft_prompt_num,
                soft_prompt_embedding = self.prompt_embedding,)
        except:
            dummy_loss = torch.tensor(0.0, device=getattr(self, 'device', 'cpu'), requires_grad=True)
            return dummy_loss.clone()
        if output_hidden_states:
            return chosen_loss, chosen_loss.hidden_states
        else:
            return chosen_loss
    
    def generate(self,inputs, use_image = True,return_full_text = False, **kwargs): 
        batch_prompts, images = self.get_formatted_prompt(inputs, use_image)

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
    
    def get_formatted_prompt(self,inputs, use_image = True, ):
        questions = inputs['question']
        if "chosen" in inputs.keys():
            answers = inputs['chosen']
        else:
            answers = None
        if not isinstance(questions,list):
            questions = [questions]
        if not isinstance(answers,list):
            answers = [answers]
        if use_image:
            assert 'image' in inputs.keys(), "Image is not provided in inputs."
            images = inputs['image'] 
            if isinstance(images,str) or isinstance(images,Image.Image) :
                images = [images]
            else:
                assert isinstance(images,list), "Image should be a list of image path or PIL.Image object."
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]

        else:
            images = None

        batch_prompts = format_prompt(self.tokenizer,questions,answers = answers ,add_generation_prompt=True,use_image=use_image)
        return batch_prompts,images
    
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
                "answer": prefix,
                "image": inputs["image"][i],
            }
            formatted_prompt, _ = self.get_formatted_prompt(prompt_inputs, use_image)
            formatted_prompt = formatted_prompt[0]
            if add_soft_prompt:
                formatted_prompt = self.add_soft_prompt(formatted_prompt, add_pos="last") 

            if safe_labels[i]:
                formatted_chosen.append(f"{formatted_prompt.strip()} {suffix.strip()}")
                formatted_rejected.append(f"{formatted_prompt.strip()} {rejected_answers[i].strip()}")
            else:
                formatted_chosen.append(f"{formatted_prompt.strip()} {chosen_answers[i].strip()}")
                formatted_rejected.append(f"{formatted_prompt.strip()} {suffix.strip()}")
           
        if use_image:
            assert 'image' in inputs.keys(), "Image is not provided in inputs."
            images = inputs['image'] 
            if isinstance(images,str) or isinstance(images,Image.Image) :
                images = [images]
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]
        return formatted_chosen, formatted_rejected, images

