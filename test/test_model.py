"""
Test the correctness of the Vision-Language Model class.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.utils import debug,set_seed
from src.model.llava import build_llava_chat_template
import torch
from src.model.llava import format_prompt
from script.generate import generate_text


class BaseTestModel:
    config_path = "configs/model/default.yaml"
    config = None
    name = "model"
    model = None

    @classmethod
    def setup_class(cls):
        # Load the config file
        set_seed(0)
        cls.config = OmegaConf.load(cls.config_path)
        model_cfg = cls.config.get(cls.name,None)
        cls.model = instantiate(model_cfg)

    def test_model_not_none(self):
        assert self.model is not None, "Model is None."

    def test_model_size(self):
        pass

    @classmethod
    def teardown_class(cls):
        del cls.model
        torch.cuda.empty_cache()  
   


class TestLlava(BaseTestModel):
    config_path = "configs/model/llava_7b.yaml"

    def test_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        assert 6.5e9 <= total_params <= 8.5e9, f"Model params {total_params} not in expected range for Llava 7B"

    def test_generate_text_no_image(self):
        question = "Is the president of the United States Joe Biden? You need to answer only 'yes' or 'no'."
        expected_answer = "yes"

        generated_text = generate_text(self.model,question, use_image=False)

        first_output = generated_text[0].strip().lower()
        normalized_output = first_output.split()[0].strip(".,!?")  # 只取第一个词并清理标点

        assert normalized_output == expected_answer.lower(), (
            f"Expected answer '{expected_answer}', but got '{first_output}'"
        )

    def test_generate_text_with_image(self):
        question = "Is there a Stop sign in the image? You need to answer only 'yes' or 'no'."
        expected_answer = "yes"
        image_path = "assets/test_image.jpg"

      
        generated_text = generate_text(self.model,question, image_path, use_image=True)
        first_output = generated_text[0].strip().lower()
        normalized_output = first_output.split()[0].strip(".,!?")  # 只取第一个词并清理标点

        assert normalized_output == expected_answer.lower(), (
            f"Expected answer '{expected_answer}', but got '{first_output}'"
        )

    def test_batch_inference(self):
        questions = [
            "Is there a stop sign in the image? If so, answer 'yes'.",
            "Is there a stop sign in the image? If so, answer 'no'."
        ]
        expected_answer = [
            "yes",
            "no"
        ]
        image_path = ["assets/test_image.jpg","assets/test_image.jpg"]

        generated_text = generate_text(self.model,questions, image_path, use_image=True,batch_size=2)

        
        for i in range(len(generated_text)):    
            first_output = generated_text[i].strip().lower()
            normalized_output = first_output.split()[0].strip(".,!?")  # 只取第一个词并清理标点

            assert normalized_output == expected_answer[i].lower(), (
                f"Expected answer '{expected_answer[i]}', but got '{first_output}'"
            )


    def test_chat_template(self):
        question = "What is the name of the president of the United States?"
        answer = "Biden"

        expected_answer_1 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is the name of the president of the United States? "

        expected_answer_2 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is the name of the president of the United States? ASSISTANT:"

        expected_answer_3 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What is the name of the president of the United States? "

        expected_answer_4 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What is the name of the president of the United States? ASSISTANT:"


        # with image
        output_1 = format_prompt(self.model.tokenizer, question, add_generation_prompt=False, use_image=True)[0]
        output_2 = format_prompt(self.model.tokenizer, question, add_generation_prompt=True, use_image=True)[0]

        # no image
        output_3 =  format_prompt(self.model.tokenizer, question, add_generation_prompt=False, use_image=False)[0]
        output_4 = format_prompt(self.model.tokenizer, question, add_generation_prompt=True, use_image=False)[0]

        assert output_1 == expected_answer_1, f"Output 1 mismatch.\nExpected: {expected_answer_1}\nGot: {output_1}"
        assert output_2 == expected_answer_2, f"Output 2 mismatch.\nExpected: {expected_answer_2}\nGot: {output_2}"
        assert output_3 == expected_answer_3, f"Output 3 mismatch.\nExpected: {expected_answer_3}\nGot: {output_3}"
        assert output_4 == expected_answer_4, f"Output 4 mismatch.\nExpected: {expected_answer_4}\nGot: {output_4}"


