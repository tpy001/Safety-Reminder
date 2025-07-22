from transformers import Qwen2VLForConditionalGeneration
from PIL import Image
from .base_model import VQAModel
from transformers import AutoProcessor, PreTrainedModel
import torch



SYSTEM_PROMPT = "You are a helpful assistant. Always answer as helpfully as possible, while being safe."
Qwen2_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

def build_qwen2_chat_template(question: str, answer = None,use_image: bool = True) -> list:
    if use_image:
        conversation = [
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
        conversation = [
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
        
    if answer is not None:
        conversation.append( 
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ]
            }
        )

    return conversation
    

def format_prompt(tokenizer, questions, answers = None,add_generation_prompt = True, use_image = True, ):
    """
    apply chat template to questions.
    return a list of prompt string
    """
    if isinstance(questions,str):
        questions = [questions]
    batch_prompts = []
    for i in range(len(questions)):
        if answers is not None and len(answers) > 0 and answers[0] is not None:
            conversation = build_qwen2_chat_template(questions[i], answers[i],use_image)
        else:
            conversation = build_qwen2_chat_template(questions[i],use_image=use_image)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=add_generation_prompt)   
        batch_prompts.append(prompt)
    return batch_prompts



class VQAQwen2VL(VQAModel):
    model_cls = Qwen2VLForConditionalGeneration
    assistant_tag = 'assistant\n'
    chat_template = Qwen2_chat_template
    add_eos_token = True

    def __init__(self, min_pixels = None,max_pixels = None,*args, **kwargs):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        super().__init__(*args, **kwargs)
        print('Load Model Successfully.')

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
    
    def get_language_model(self):
        return self.model.model
    
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


    def get_labels(self,prompts,substring,return_tensor=True):
        def find_sublist_position(main_list, sub_list):
            for i in range(len(main_list) - len(sub_list) + 1):
                # 检查从当前位置i开始的子列表是否与sub_list相同
                if main_list[i:i+len(sub_list)] == sub_list:
                    return i + len(sub_list)   # 返回子列表结束位置的下一个位置
            raise ValueError("子列表未在主列表中找到。")
        
        labels = []
        # substring = 'ASSISTANT:'
        # substring_ids = self.processor.tokenizer.encode(substring)[1:]
        substring_ids = self.processor.tokenizer.encode(substring) # Here is the difference
        for prompt in prompts:
            prompt = prompt.tolist()
            position = find_sublist_position(prompt, substring_ids)
            label  = [-100] * (position) + prompt[position:]
            labels.append(label)
        if return_tensor:
            return torch.tensor(labels)
        else:
            return labels


    