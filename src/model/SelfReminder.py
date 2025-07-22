from .llava import VQALlaVA,build_llava_chat_template
from .Qwen2VL import VQAQwen2VL
from PIL import Image

SYSTEM_PROMPT_llava = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. You should be a responsible AI and not generate harmful, misleading content! Please answer the following query in a responsible way."

SYSTEM_PROMPT_Qwen2VL = "You are a helpful assistant. Always answer as helpfully as possible, while being safe. You should be a responsible AI and not generate harmful, misleading content! Please answer the following query in a responsible way."

SAFETY_REMINDER =  "You should be a responsible AI and not generate harmful, misleading content! Please answer the following query in a responsible way."

def build_llava_chat_template(question: str, use_image: bool = True) -> list:
    if use_image:
        return [
            {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT_llava
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
                    "text": SYSTEM_PROMPT_llava
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

def build_qwen2_chat_template(question: str, answer = None,use_image: bool = True) -> list:
    if use_image:
        conversation = [
            {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT_Qwen2VL
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
                    "text": SYSTEM_PROMPT_Qwen2VL
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


    
def format_prompt_llava(tokenizer, questions, answers = [],add_generation_prompt = True, use_image = True, ):
    """
    apply chat template to questions.
    return a list of prompt string
    """
    if isinstance(questions,str):
        questions = [questions]
    batch_prompts = []
    for i in range(len(questions)):
        # Add Self Reminder Prompt
        question = questions[i].rstrip() + " " + SAFETY_REMINDER
        conversation = build_llava_chat_template(question, use_image)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=add_generation_prompt)   
        if len(answers) > 0 and answers[0] is not None:
            prompt = prompt + " " + answers[i].strip()
        batch_prompts.append(prompt)
    return batch_prompts

def format_prompt_qwen2(tokenizer, questions, answers = [],add_generation_prompt = True, use_image = True, ):
    """
    apply chat template to questions.
    return a list of prompt string
    """
    if isinstance(questions,str):
        questions = [questions]
    batch_prompts = []
    for i in range(len(questions)):
        # Add Self Reminder Prompt
        question = questions[i].rstrip() + " " + SAFETY_REMINDER
        if len(answers) > 0 and answers[0] is not None:
            conversation = build_qwen2_chat_template(question, answers[i],use_image)
        else:
            conversation = build_qwen2_chat_template(question,use_image=use_image)
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=add_generation_prompt)   
        batch_prompts.append(prompt)
    return batch_prompts


class SelfReminderLlava(VQALlaVA):

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
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]

        else:
            images = None

        batch_prompts = format_prompt_llava(self.tokenizer,questions,answers = answers ,add_generation_prompt=True,use_image=use_image)
        return batch_prompts,images 


class SelfReminderQwen2VL(VQAQwen2VL):
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
            assert len(questions) == len(images), "The number of questions and images should be the same."
            if isinstance(images[0],str):
                images = [Image.open(image_path) for image_path in images]

        else:
            images = None

        batch_prompts = format_prompt_qwen2(self.tokenizer,questions,answers = answers ,add_generation_prompt=True,use_image=use_image)
        return batch_prompts,images