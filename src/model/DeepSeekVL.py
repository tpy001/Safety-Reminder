from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from PIL import Image
from .base_model import VQAModel
from deepseek_vl.utils.io import load_pil_images
import torch
from typing import List, Dict
from PIL import Image
from transformers.generation import GenerationConfig


# SYSTEM_PROMPT = "You are a helpful assistant. Always answer as helpfully as possible, while being safe."

def build_DeepSeekVL_chat_template(question: str, answer = None,image = None) -> list:
    if image is not None:
        return [
            {
                "role": "User",
                "content": "<image_placeholder>" + question,
                # "images": [image]
            },
            {
                "role": "Assistant",
                "content": "" if answer is None else answer,
            }
    ]
    else:
      return [
             {
                "role": "User",
                "content": question,
            },
            {
                "role": "Assistant",
                "content": "" if answer is None else answer,
            },
        ]

def format_prompt(vl_chat_processor, questions, answers = None,add_generation_prompt = True, image_paths=None, add_eos_token = False):
    """
    apply chat template to questions.
    return a list of prompt string
    """
    if isinstance(questions,str):
        questions = [questions]
    formatted_prompt_list = []
    for i in range(len(questions)):
        if image_paths is not None:
            conversation = build_DeepSeekVL_chat_template(question = questions[i],image=image_paths[i])
        else:
            conversation = build_DeepSeekVL_chat_template(question = questions[i])

        formatted_prompt  = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            system_prompt=vl_chat_processor.system_prompt
        )
        if answers is not None and len(answers) > 0:
            formatted_prompt = formatted_prompt.strip() + " " + answers[i]
            if add_eos_token:
                formatted_prompt = formatted_prompt + "<｜end▁of▁sentence｜>"

        formatted_prompt_list.append(formatted_prompt)
    return formatted_prompt_list



class VQADeepSeek(VQAModel):
    model_cls = AutoModelForCausalLM
    assistant_tag = "Assistant:"

    def forward(self,inputs,use_image=True,output_hidden_states=False, use_answer = True, **kwargs):
        self.check_inputs(inputs,use_image,use_answer)
        if not use_answer and "chosen" in inputs.keys():
            inputs.pop("chosen")
        formatted_prompt,images = self.get_formatted_prompt(inputs,use_image, **kwargs)
        return self._forward(formatted_prompt,images=images,output_hidden_states=output_hidden_states, **kwargs)
    
    def _forward(self, formatted_prompt, images=None, output_hidden_states=False, **kwargs):
        prepare_input_list = []
        for i in range(len(formatted_prompt)):
            if isinstance(images[i],Image.Image):
                images[i] = images[i].convert("RGB")
            prepare_input = self.processor(prompt=formatted_prompt[i],images = [images[i]],force_batchify=False)
            prepare_input_list.append(prepare_input)
        batch_inputs = self.processor.batchify(prepare_input_list).to(self.model.device)

        label_ids = self.get_labels(batch_inputs['input_ids'],substring = self.assistant_tag).to(self.device)

        image_token_id = self.tokenizer.encode("<image_placeholder>")[1]
        
        batch_inputs['input_ids'], batch_inputs['attention_mask'], label_ids = self.truncate_around_image_token(
            input_ids=batch_inputs['input_ids'],
            attention_mask=batch_inputs['attention_mask'],
            label_ids=label_ids,
            max_length=self.max_context_length,
            image_token_id=image_token_id,
        )

        if self.trainable== "frozen":
            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
        else:
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)

        outputs = self.model.language_model(
            inputs_embeds = inputs_embeds,
            attention_mask = batch_inputs['attention_mask'],
            return_dict=True,
            labels = label_ids,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        if output_hidden_states:
            return outputs.loss, outputs.hidden_states
        else:
            return outputs.loss
        
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

        # processor = AutoProcessor.from_pretrained(self.model_path)
        processor = VLChatProcessor.from_pretrained(self.model_path)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        self.processor = processor
        self.tokenizer = processor.tokenizer

        # if self.tokenizer.chat_template is None:
        #     assert self.chat_template is not None, "chat_template is not provided in tokenizer."
        #     self.tokenizer.chat_template = self.chat_template
        return model
    
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
                images = [Image.open(image_path).convert("RGB") for image_path in images]

        else:
            images = None

        if use_image:
            formatted_prompt = format_prompt(self.processor,questions,answers = answers ,add_generation_prompt=True,image_paths=images)
        else:
            formatted_prompt = format_prompt(self.processor,questions,answers = answers ,add_generation_prompt=True)
        return formatted_prompt,images
    
    def generate(self,inputs, use_image = True, **kwargs):  
        format_prompt,images = self.get_formatted_prompt(inputs, use_image)
        return self.generate_text(format_prompt,images,use_image = use_image, **kwargs)
    
    def train(self,mode=True):
        # if self.trainable is None:
        #     raise ValueError("trainable should be set to 'all', 'visual_encoder', or 'language_model' in config file.")
        # elif self.trainable == 'all':
        #     self.model.train()
        # elif self.trainable == 'visual_encoder':
        #     self.model.visual_encoder.train()
        # elif self.trainable == 'language_model':
        #     self.model.language_model.train()
        # elif self.trainable == 'frozen':
        #     self.model.eval()
        self.model.eval()

    def generate_text(self,format_prompt, images, use_image = True, return_full_text=False,**kwargs):
        responses = []
        prepare_input_list = []
        use_image = False # debug only
        for i in range(len(format_prompt)):
            if use_image:
                prepare_input = self.processor(prompt=format_prompt[i],images = [images[i].convert("RGB")],force_batchify=False)
                prepare_input_list.append(prepare_input)
            else:
                prepare_input = self.processor(prompt=format_prompt[i],force_batchify=False)
                prepare_input_list.append(prepare_input)

        batch_inputs = self.processor.batchify(prepare_input_list)
        batch_inputs = batch_inputs.to(self.model.device)

        if use_image:
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
        else:
            inputs_embeds = self.model.language_model.get_input_embeddings()(batch_inputs["input_ids"])
        generate_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.generate_config["max_new_tokens"],
                do_sample=self.generate_config["do_sample"],
                use_cache=self.generate_config["use_cache"]
        )
           
        generated_text = self.processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        for j in range(len(generate_ids)):
            responses.append(generated_text[j].strip(" "))
        return responses