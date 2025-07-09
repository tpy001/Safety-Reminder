import torch
from transformers import AutoProcessor, PreTrainedModel
from transformers.generation import GenerationConfig

class VQAModel(torch.nn.Module):
    model_cls = PreTrainedModel
    assistant_tag = "ASSISTANT:"
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
        self.model = self.load_model()

    def load_model(self):
        if self.device not in ['cpu', 'cuda', 'auto', None]:
            raise ValueError("device should be 'cpu', 'cuda', 'auto', or None'.")

        if self.device is None or self.device == 'auto':
            model = self.model_cls.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, device_map='auto'
            )
            self.device = model.device
        else:
            model = self.model_cls.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, device_map=self.device
            )

        processor = AutoProcessor.from_pretrained(self.model_path)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        self.processor = processor
        self.tokenizer = processor.tokenizer
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
            prompt = prompt.tolist()
            position = find_sublist_position(prompt, substring_ids)
            label  = [-100] * (position) + prompt[position:]
            labels.append(label)
        if return_tensor:
            return torch.tensor(labels)
        else:
            return labels
    
    def _forward(self, formatted_prompt, images=None, output_hidden_states=False):
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        label_ids = self.get_labels(inputs['input_ids'],substring = self.assistant_tag).to(self.device)
        max_seq_length = self.max_context_length

        current_seq_length = inputs['input_ids'].shape[1]
        if current_seq_length > max_seq_length:
            inputs['input_ids'] = inputs['input_ids'][:, :max_seq_length]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_seq_length]
            label_ids = label_ids[:, :max_seq_length]

        outputs = self.model(
            **inputs,
            return_dict=True,
            labels = label_ids,
            output_hidden_states=output_hidden_states
        )
        if output_hidden_states:
            return outputs.hidden_states
        else:
            return outputs.loss

    def check_inputs(self,inputs,use_image=True):
        assert "question" in inputs.keys(), "Question is not provided in inputs."
        assert "chosen" in inputs.keys(), "Chosen answer is not provided in inputs."
        if use_image:
            assert "image" in inputs.keys(), "Image is not provided in inputs."

    def forward(self,inputs,use_image=True,output_hidden_states=False):
        self.check_inputs(inputs,use_image)
        formatted_prompt = self.formatted_prompt(inputs,use_image)
        return self._forward(formatted_prompt,images=inputs.get("image",None),output_hidden_states=output_hidden_states)
    
    def get_formatted_prompt(self, inputs,use_image=True):
        raise NotImplementedError("formatted_prompt method is not implemented.")
        
    def generate(self,inputs, use_image = True, **kwargs):  
        formatted_prompt, images = self.get_formatted_prompt(inputs, use_image)
        return self.generate_text(formatted_prompt, images=images,**kwargs)
    

    def generate_text(self,formatted_prompt, images=None,return_full_text=False, **kwargs):
        responses = []
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, generation_config = GenerationConfig(**self.generate_config), **kwargs)

        if not return_full_text:
            length = inputs['input_ids'].shape[-1]
            generate_ids = generate_ids[:, length:]
        generated_text = self.processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        for j in range(len(generate_ids)):
            responses.append(generated_text[j].strip(" "))
        return responses

