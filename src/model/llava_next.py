from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from .base_model import VQAModel


SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
LlaVA_chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

def build_llava_chat_template(question: str, use_image: bool = True) -> list:
    if use_image:
        return [
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
        if answers is not None and len(answers) > 0 and answers[0] is not None:
            prompt = prompt + " " + answers[i].strip()
        batch_prompts.append(prompt)
    return batch_prompts



class VQALlaVA_Next(VQAModel):
    model_cls = LlavaNextForConditionalGeneration
    assistant_tag = "[/INST]"
    chat_template = LlaVA_chat_template

    def get_language_model(self):
        return self.model.language_model
    
    def get_formatted_prompt(self,inputs, use_image = True, ):
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

        batch_prompts = format_prompt(self.tokenizer,questions,answers = answers ,add_generation_prompt=True,use_image=use_image)
        return batch_prompts,images
    
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