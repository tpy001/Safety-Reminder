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
from .prompt_tuning import PromptTuningLlava
import random
from typing import Union, Tuple,List,Optional,Any
from copy import deepcopy


def remove_after_last_keyword(text: str, keyword: str = "like") -> str:
    idx = text.rfind(f"{keyword}")  # 找到最后一个 ' like ' 的起始位置
    if idx == -1:
        return text  # 如果没有 like，就原样返回
    return text[:idx+len(keyword)].strip()  # 截断并去掉尾部空格

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

    
class SAPTLlava(PromptTuningLlava):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = {
            "soft_prompt_num": self.soft_prompt_num,
            "soft_prompt_id": self.soft_prompt_id,
            "soft_prompt_text": self.soft_prompt_text,
        }
        self.modify_sample_func(SAPTLlava_sample,params)

    def modify_sample_func(self, custom_sample_func, params):
        """
        Replace the model.model.sample function with a custom one.
        """

        target = self.model.model

        # The original sample method is cached for easy restoration
        if not hasattr(target, "_original_sample"):
            target._original_sample = self.model.model._sample

        # Inject custom parameters as member variables
        for k, v in params.items():
            setattr(target, k, v)

        setattr(target,"tokenizer", self.model.tokenizer)

        # Bind custom_sample_func to the model.model instance
        bound_sample = custom_sample_func.__get__(self.model.model, type(self.model.model))

        # Setting up a new sample method
        target._sample = bound_sample
      
        return self.model


    def get_labels(self, prompts, return_tensor=True):
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
        batch_prompts_chosen, batch_prompts_rejected, images = self.get_formated_prompt(inputs, use_image,add_soft_prompt=True)
        try:
            chosen_loss = self._forward(batch_prompts_chosen, images=images,output_hidden_states=output_hidden_states)
        except:
            logger.info("Error in forward function of SAPT")
            dummy_loss = torch.tensor(0.0, device=getattr(self, 'device', 'cpu'), requires_grad=True)
            return dummy_loss.clone()
        if output_hidden_states:
            return chosen_loss, chosen_loss.hidden_states
        else:
            return chosen_loss
    
    def _forward(self, formatted_prompt, images=None, output_hidden_states=False):
        """
        Forward method to compute either per-sample loss or hidden states.

        Args:
            formatted_prompt (List[str]): Prompt strings with soft prompts included.
            images (Optional): Associated images for multi-modal input.
            output_hidden_states (bool): If True, return hidden states instead of loss.

        Returns:
            Union[Tensor, List[Tensor]]:
                - If output_hidden_states: return hidden states
                - Else: return per-sample NLL loss (Tensor of shape [batch_size])
        """
        # Tokenization
        inputs = self.model.processor(
            images=images,
            text=formatted_prompt,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Get labels with masking
        label_ids = self.get_labels(
            prompts=inputs['input_ids']
        ).to(self.model.device)

        # Disable averaging in the loss computation
        outputs = self.model.model(
            **inputs,
            return_dict=True,
            labels=label_ids,
            output_hidden_states=output_hidden_states
        )

        if output_hidden_states:
            return outputs.loss, outputs.hidden_states
        else:
            return outputs.loss



    def generate(self,inputs, use_image = True, **kwargs):  
        # formatted_prompt_chosen,formatted_prompt_rejected, images = self.get_formated_prompt(inputs, use_image,add_soft_prompt=True)
        # for i in range(len(formatted_prompt_chosen)):
        #     formatted_prompt_chosen[i] = remove_after_last_keyword(formatted_prompt_chosen[i],keyword=self.soft_prompt_text)
        #     formatted_prompt_rejected[i] = remove_after_last_keyword(formatted_prompt_rejected[i],keyword=self.soft_prompt_text)
        return self.model.generate(inputs, use_image, **kwargs)
    
    
    def get_formated_prompt(
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
            formatted_prompt, _ = self.model.get_formated_prompt(prompt_inputs, use_image)
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


from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList  
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)
from transformers.cache_utils import *


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
    token_strings = tokenizer.convert_ids_to_tokens(token_ids.tolist())

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


def validate_is_safe(self,model_inputs,soft_prompt_id,soft_prompt_num,logits_processor):
    """
        Validate if the current generations are safe or not using our trained safety prompt
    """
    batch_size = model_inputs["input_ids"].shape[0]
    new_model_inputs = deepcopy(model_inputs)
    input_ids =  new_model_inputs["input_ids"]
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
    new_model_inputs.pop("past_key_values")

    with torch.no_grad():
        outputs = self(**new_model_inputs, return_dict=True)

    next_token_logits = outputs.logits.clone()[:, -1, :].float()
    next_token_logits = next_token_logits.to(input_ids.device)

    # pre-process distribution
    next_token_scores = logits_processor(input_ids, next_token_logits)

    next_tokens = torch.argmax(next_token_scores, dim=-1)

    is_capital_list = check_if_tokens_start_with_capital(next_tokens, self.tokenizer)

    is_safe = ~is_capital_list

    return is_safe
    

def SAPTLlava_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
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

        interval = 16
        count = 0
        is_safe = torch.tensor( [True] * (batch_size)).to(input_ids.device)
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
           
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if count % interval == 1:
                safe = validate_is_safe(self,model_inputs,self.soft_prompt_id,self.soft_prompt_num,logits_processor)
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