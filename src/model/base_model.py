import torch
from transformers import AutoProcessor, PreTrainedModel
from transformers.generation import GenerationConfig

class VQAModel(torch.nn.Module):
    model_cls = PreTrainedModel
    assistant_tag = "ASSISTANT:"
    chat_template = None
    def __init__(self, model_path, torch_dtype=torch.float16, device='auto', max_context_length=1024,trainable = "frozen",generate_config = {},*args, **kwargs):
        """
        Args:
            model_path (str): HuggingFace model path or local directory
            model_cls (Type): Model class (e.g., LlavaForConditionalGeneration)
            torch_dtype (torch.dtype): dtype for model weights
            device (str): 'cpu', 'cuda', or 'auto'
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.trainable = trainable
        self.generate_config = generate_config
        self.model = self.load_model(*args, **kwargs)

    def load_model(self,*args, **kwargs):
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

        processor = AutoProcessor.from_pretrained(self.model_path)
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
        raise NotImplementedError("train method is not implemented.")

    def get_labels(self,prompts,substring,return_tensor=True):
        def find_sublist_position(main_list, sub_list):
            for i in range(len(main_list) - len(sub_list) + 1):
                # 检查从当前位置i开始的子列表是否与sub_list相同
                if main_list[i:i+len(sub_list)] == sub_list:
                    return i + len(sub_list)   # 返回子列表结束位置的下一个位置
            raise ValueError("子列表未在主列表中找到。")
        
        labels = []
        # substring = 'ASSISTANT:'
        substring_ids = self.processor.tokenizer.encode(substring)[1:]
        for prompt in prompts:
            if not isinstance(prompt,list):
                prompt = prompt.tolist()
            position = find_sublist_position(prompt, substring_ids)
            label  = [-100] * (position) + prompt[position:]
            labels.append(label)
        if return_tensor:
            return torch.tensor(labels)
        else:
            return labels
    
    def truncate_around_image_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        max_length: int,
        image_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Truncates input sequences while preserving all image tokens.
        Tries front, back, and center truncation strategies in that order.
        
        Args:
            input_ids: Input tensor of shape (batch_size, seq_length)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)
            label_ids: Label tensor of shape (batch_size, seq_length)
            max_length: Maximum allowed sequence length
            image_token_id: ID of the image token to preserve
        
        Returns:
            Tuple of truncated (input_ids, attention_mask, label_ids)
        
        Raises:
            ValueError: If no image tokens found or max_length is too small
        """
        batch_size, current_length = input_ids.shape
        
        # Return unchanged if sequence is short enough
        if current_length <= max_length:
            return input_ids, attention_mask, label_ids

        # Count image tokens
        image_mask = input_ids[0] == image_token_id
        num_images = image_mask.sum().item()
        
        if num_images == 0:
            raise ValueError("No image tokens detected in input_ids")
        
        if max_length < num_images:
            raise ValueError(f"max_length ({max_length}) too small to accommodate {num_images} image tokens")

        text_budget = max_length - num_images
        if text_budget <= 0:
            raise ValueError(f"max_length ({max_length}) too small to include any text tokens")

        def has_all_images(slice_indices: slice) -> bool:
            """Check if slice contains all image tokens."""
            return (input_ids[0, slice_indices] == image_token_id).sum().item() == num_images

        # Try 1: Front truncation
        front_slice = slice(0, max_length)
        if has_all_images(front_slice):
            return (
                input_ids[:, front_slice],
                attention_mask[:, front_slice],
                label_ids[:, front_slice]
            )

        # Try 2: Back truncation
        back_slice = slice(-max_length, None)
        if has_all_images(back_slice):
            return (
                input_ids[:, back_slice],
                attention_mask[:, back_slice],
                label_ids[:, back_slice]
            )

        # Try 3: Center truncation
        # Keep all image tokens and distribute remaining budget around them
        image_indices = torch.where(image_mask)[0]
        leftmost_image = image_indices[0].item()
        rightmost_image = image_indices[-1].item()
        
        # Calculate available space for text on both sides
        text_space = text_budget // 2
        left_bound = max(0, leftmost_image - text_space)
        right_bound = min(current_length, rightmost_image + text_space + 1)
        
        # Adjust if we couldn't get enough space on right
        if right_bound - left_bound < max_length:
            left_bound = max(0, right_bound - max_length)
        
        center_slice = slice(left_bound, right_bound)
        
        return (
            input_ids[:, center_slice],
            attention_mask[:, center_slice],
            label_ids[:, center_slice]
        )

    def _forward(self, formatted_prompt, images=None, output_hidden_states=False, **kwargs):
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        label_ids = self.get_labels(inputs['input_ids'],substring = self.assistant_tag).to(self.device)

        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        
        inputs['input_ids'], inputs['attention_mask'], label_ids = self.truncate_around_image_token(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            label_ids=label_ids,
            max_length=self.max_context_length,
            image_token_id=image_token_id,
        )


        outputs = self.model(
            **inputs,
            return_dict=True,
            labels = label_ids,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        if output_hidden_states:
            return outputs.hidden_states
        else:
            return outputs.loss

    def check_inputs(self,inputs,use_image=True,use_answer=True):
        assert "question" in inputs.keys(), "Question is not provided in inputs."
        if use_answer:
            assert "chosen" in inputs.keys(), "Chosen answer is not provided in inputs."
        if use_image:
            assert "image" in inputs.keys(), "Image is not provided in inputs."

    def forward(self,inputs,use_image=True,output_hidden_states=False, use_answer = True, **kwargs):
        self.check_inputs(inputs,use_image,use_answer)
        if not use_answer:
            inputs.pop("chosen")
        formatted_prompt,images = self.get_formatted_prompt(inputs,use_image)
        return self._forward(formatted_prompt,images=images,output_hidden_states=output_hidden_states, **kwargs)
    
    def get_formatted_prompt(self, inputs,use_image=True):
        raise NotImplementedError("formatted_prompt method is not implemented.")
        
    def generate(self,inputs, use_image = True, **kwargs):  
        formatted_prompt, images = self.get_formatted_prompt(inputs, use_image)
        return self.generate_text(formatted_prompt, images=images, **kwargs)
    

    def generate_text(self,formatted_prompt, images=None,return_full_text=False,**kwargs):
        responses = []
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, generation_config = GenerationConfig(**self.generate_config),**kwargs)

        if not return_full_text:
            length = inputs['input_ids'].shape[-1]
            generate_ids = generate_ids[:, length:]
        generated_text = self.processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        for j in range(len(generate_ids)):
            responses.append(generated_text[j].strip(" "))
        return responses

