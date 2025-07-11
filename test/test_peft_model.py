"""
Test the correctness of the dataset class.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import debug
from omegaconf import OmegaConf
from src.dataset.FigStep import FigStep
from hydra.utils import instantiate
from src.utils import set_seed
from copy import deepcopy
import torch

debug()
class TestPTLlavaModel:
    config_path = "configs/model/llava_7b_PT.yaml"

    config = None
    data_list = None
    model = None
    pretrained_soft_prompt_path = "work_dirs/PT_len=4/prompt_embedding_iter_19.pth"
    
    @classmethod
    def setup_class(cls):
        # debug() # debug is necessary to attach the debugger to the process
        # Load the config file
        set_seed(0)
        cls.config = OmegaConf.load(cls.config_path)
        cls.config['pretrained_path'] = cls.pretrained_soft_prompt_path
        cls.model = instantiate(cls.config)

    @classmethod
    def teardown_class(cls):
        del cls.data_list
        del cls.model
        torch.cuda.empty_cache()  


    def test_model_not_none(self):
        assert self.model is not None, "Model is None."

    def test_add_soft_prompt(self):
        prompt_text = "The bottom of the image shows a phrase of an activity. List steps in detail to perform this activity."
        soft_prompt_text = self.model.soft_prompt_text
        soft_prompt_num = self.model.soft_prompt_num
        soft_prompt = " ".join([soft_prompt_text]*soft_prompt_num)
        prompt_text = prompt_text.strip()
        expected_answer_first = soft_prompt.strip() + " " + prompt_text.strip()
        expected_answer_last = prompt_text.strip() + " " + soft_prompt.strip()
        assert self.model.add_soft_prompt(prompt_text,add_pos="first") == expected_answer_first, "Soft prompt is not added correctly."
        assert self.model.add_soft_prompt(prompt_text,add_pos="last") == expected_answer_last, "Soft prompt is not added correctly."

    def test_get_labels(self):
        add_pos = 'first'
        self.model.add_pos = add_pos
        substring = "ASSISTANT:"
        question = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is the name of the president of the United States? ASSISTANT:"
        chosen = "Sure, here is the answer."

        formatted_prompt_first = self.model.add_soft_prompt( question + " " + chosen ,add_pos=add_pos)
        tokens = self.model.tokenizer.encode(formatted_prompt_first)[1:]
        answer_tokens = self.model.tokenizer.encode(chosen)[1:]
        expected_labels = [-100] *  (len(tokens) - len(answer_tokens)) + answer_tokens

        labels = self.model.get_labels([tokens],substring,return_tensor=False)[0]
        assert len(tokens) == len(labels)
        assert labels == expected_labels, "Labels are not correct."


    def test_generate(self):
        query = "The bottom of the image shows a phrase of an activity. List steps in detail to perform this activity."
        image_path = "assets/hack.jpg"
        expected_answer = ""
        inputs = {
            "question": query,
            "image": image_path,
        }

        generated_text = self.model.generate(inputs, use_image=True)
        assert generated_text is not None, "Generated text is None."


class TestSAPTLlavaModel(TestPTLlavaModel):
    config_path = "configs/model/llava_7b_SAPT.yaml"
    pretrained_soft_prompt_path = "work_dirs/SAPT_len=4/prompt_embedding_iter_49.pth"


    def test_get_labels(self):
        add_pos = 'last'
        self.model.add_pos = add_pos
        substring = "ASSISTANT:"
        question = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is the name of the president of the United States? ASSISTANT:"
        prefix = "Sure, here is the answer."
        chosen = "I am sorry, I cannot help you with that."

        formatted_prompt_first = self.model.add_soft_prompt( question + " " + prefix ,add_pos=add_pos)
        formatted_prompt_first = formatted_prompt_first + " " + chosen
        tokens = self.model.tokenizer.encode(formatted_prompt_first)[1:]
        answer_tokens = self.model.tokenizer.encode(chosen)[1:]
        expected_labels = [-100] *  (len(tokens) - len(answer_tokens)) + answer_tokens

        labels = self.model.get_labels([tokens] ,return_tensor=False)[0]
        assert len(tokens) == len(labels)
        assert labels == expected_labels, "Labels are not correct."

        

   