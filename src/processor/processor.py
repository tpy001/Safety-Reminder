from PIL import Image
separators = [".", ",", "!", "?"]  
from copy import deepcopy

class TextPreprocessor:
    def __init__(self, system_prompt='',chat_template={}):
        self.system_prompt = system_prompt
        self.chat_template = chat_template

    def preprocess_prompt(self,prompts):  
        # 1. remove the last space of input prompts
        # 2. add . to the end of input prompts if there is no separator in the end

        # input:
        #        prompts:  a list of prompts
        # output:
        #        prompts after preprocessing

        for i in range(len(prompts)):
            prompts[i] = prompts[i].strip(" ")
            if not any(prompts[i].endswith(separator) for separator in separators):
                prompts[i] += "."
        return prompts

    def __call__(self,prompts,tokenizer):
        prompts = self.preprocess_prompt(prompts)
        res = []
        for i in range(len(prompts)):
            chat_template = deepcopy(self.chat_template)
            for j in range(len(chat_template)):
                if chat_template[j]['role'] == "system":
                    chat_template[j]['content'] = self.system_prompt
                elif chat_template[j]['role'] == "user":
                    chat_template[j]['content'] = prompts[i]
            formatted_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
            formatted_prompt = formatted_prompt.strip(" ")
            res.append(formatted_prompt)
        return res
    
class TextImagePreprocessor(TextPreprocessor):
    def __init__(self, system_prompt='',chat_template={}):
        super().__init__(system_prompt,chat_template)

    def load_image(self,image_path):
        image = Image.open(image_path)
        return image

    def __call__(self,prompts,images,tokenizer,answers=None):
        prompts = self.preprocess_prompt(prompts)
        if isinstance(images[0],str):
            images = [self.load_image(image_path) for image_path in images]

        text = []
        for i in range(len(prompts)):
            flag = 0
            chat_template = deepcopy(self.chat_template)
            for j in range(len(chat_template)):
                if chat_template[j]['role'] == "system":
                    chat_template[j]['content'][0]['text'] = self.system_prompt
                elif chat_template[j]['role'] == "user":
                    content = chat_template[j]['content']
                    if 'text' in content[0].keys():
                        content[0]['text'] = prompts[i]
                    elif 'text' in content[1].keys():
                        content[1]['text'] = prompts[i]
                elif chat_template[j]['role'] == "assistant":
                    flag = 1
                    if answers is not None:
                        chat_template[j]['content'][0]['text'] = answers[i]

            if flag == 0:
                formatted_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False,add_generation_prompt=True)
            else:
                formatted_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
            formatted_prompt = formatted_prompt.strip(" ")
            text.append(formatted_prompt)
        return text,images
    
class MiniGPT4Preprocessor(TextImagePreprocessor):
    def __init__(self, has_image = True,image_size=(224,224),patch_size=14,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_image = has_image
        self.image_size = image_size
        self.patch_size = patch_size

    def __call__(self,prompts,images,tokenizer,answers=None):
        prompts = self.preprocess_prompt(prompts)
        if isinstance(images[0],str):
            images = [self.load_image(image_path) for image_path in images]

        if self.has_image:
            assert self.image_size[0] % self.patch_size == 0 and self.image_size[1] % self.patch_size == 0, "image size should be divisible by patch size"
            image_token_nums = self.image_size[0]//self.patch_size * self.image_size[1]//self.patch_size
            image_token_nums = image_token_nums // 4 # In the minigptv2 paper, 4 adhere visual token will be merged to one token
            image_token = "<Img>" + "<ImageHere>"*image_token_nums + "</Img> "
            prompts = [ image_token + prompt  for prompt in prompts]
        text = []
        for i in range(len(prompts)):
            chat_template = deepcopy(self.chat_template)
            for j in range(len(chat_template)):
                if chat_template[j]['role'] == "system":
                    chat_template[j]['content'] = self.system_prompt
                elif chat_template[j]['role'] == "user":
                    chat_template[j]['content'] = prompts[i]
            formatted_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
            formatted_prompt = formatted_prompt.strip(" ")
            text.append(formatted_prompt)
        return text,images