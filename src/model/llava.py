from transformers import AutoProcessor,LlavaForConditionalGeneration
from .base_model import BaseModel
import torch
from transformers.generation import GenerationConfig
from PIL import Image


SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

def build_llava_chat_template(question: str, use_image: bool = True) -> list:
    if use_image:
        return [
            {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]
        }
    ]
    else:
        return [
            {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ]
        }
    ]

def format_prompt(tokenizer, questions, answers = None,add_generation_prompt = True, use_image = True, ):
    """
    apply chat template to questions.
    return a list of prompt string
    """
    if isinstance(questions,str):
        questions = [questions]
    batch_prompts = []
    for i in range(len(questions)):
        conversation = build_llava_chat_template(questions[i], use_image)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=add_generation_prompt)   
        if answers is not None:
            prompt = prompt + " " + answers[i].strip()
        batch_prompts.append(prompt)
    return batch_prompts

    
class LlaVA(BaseModel):
    def __init__(self, generate_config,trainable='all',assistant_tag = None,to_device=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_config = generate_config
        self.trainable = trainable
        self.assistant_tag = assistant_tag

        self.load_model(self.device)
        self.model.gradient_checkpointing_enable()
        print('Load Model Successfully.')

    def get_language_model(self):
        return self.model.language_model
    
    def set_language_model(self, model):
        self.model.language_model = model
    
    def set_language_model(self, model):
        self.model.language_model = model
    
    def load_model(self,device=None):
        if device is None:
            self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype = self.torch_dtype, device_map='auto')
        else:
            if device == 'cpu':
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype = self.torch_dtype, device_map='cpu')
            elif device == 'cuda':
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype = self.torch_dtype, device_map='cuda')
            elif device == 'auto':
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype = self.torch_dtype, device_map='auto')
                self.device = self.model.device
            else:
                raise ValueError("device should be 'cpu' or 'cuda'.")
        processor = AutoProcessor.from_pretrained(self.model_path)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
        self.processor = processor
        self.tokenizer = self.processor.tokenizer

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

    def _forward(self,formatted_prompt, images=None, output_hidden_states=False):
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        # By default, the text following the assistant tag is treated as the target answer.
        label_ids = self.get_labels(inputs['input_ids'],substring = self.assistant_tag).to(self.device)
        outputs = self.model(
            **inputs,
            return_dict=True,
            labels = label_ids,
            output_hidden_states=output_hidden_states
        )
        if output_hidden_states:
            return outputs.hidden_states[-1]
        else:
            return outputs.loss
        
    def forward(self, inputs,use_image = True, output_hidden_states=False):
        formatted_prompt, images = self.get_formated_prompt(inputs, use_image)
        return self._forward(formatted_prompt, images=images, output_hidden_states=output_hidden_states)
   

    def get_formated_prompt(self,inputs, use_image = True, ):
        questions = inputs['question']
        if "answer" in inputs.keys():
            answers = inputs['answer']
        else:
            answers = None
        if not isinstance(questions,list):
            questions = [questions]
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

        batch_prompts = format_prompt(self.tokenizer,questions,answers = answers ,add_generation_prompt=True,use_image=use_image)
        return batch_prompts,images

    def generate(self,inputs, use_image = True, **kwargs):  
        formatted_prompt, images = self.get_formated_prompt(inputs, use_image)
        return self.generate_text(formatted_prompt, images=images)
    
    def generate_text(self,formatted_prompt, images=None):
        responses = []
        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, generation_config = GenerationConfig(**self.generate_config))

        if not self.generate_config.return_full_text:
            length = inputs['input_ids'].shape[-1]
            generate_ids = generate_ids[:, length:]
        generated_text = self.processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        for j in range(len(generate_ids)):
            responses.append(generated_text[j].strip(" "))
        return responses
    

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
    