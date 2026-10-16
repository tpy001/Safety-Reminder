"""Single-request inference. Model weights are loaded once per instance."""

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, LlavaForConditionalGeneration
from transformers import LogitsProcessorList, StoppingCriteriaList

from .sapt import SafetyReminder
from .config import MODEL_PATH, RESOURCES


SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


class LlavaInference:
    def __init__(self, mode="baseline", model_path=MODEL_PATH, device="cuda:0",
                 max_new_tokens=256, prompt_path=None, classifier_path=None):
        if mode not in ("baseline", "sapt"):
            raise ValueError("mode must be baseline or sapt")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.chat_template = CHAT_TEMPLATE
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # LLaVA maps vision-encoder features through a projector into the language decoder.
        # Keep this backbone intact; SAPT only replaces four token embeddings via a local hook.
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device, local_files_only=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = self.model.get_input_embeddings().weight.device
        self.reminder = None
        if mode == "sapt":
            self.reminder = SafetyReminder(
                self.model, self.tokenizer,
                prompt_path or RESOURCES / "prompt_embedding_iter_21.pth",
                classifier_path or RESOURCES / "llava_pca_cls.npz")
        self.generation_config = GenerationConfig(
            bos_token_id=1, eos_token_id=2, pad_token_id=32001,
            do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)

    def _format(self, question, has_image):
        content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": question}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _inputs(self, text, image, reserve_generation=True):
        inputs = self.processor(text=[text], images=[image] if image is not None else None,
                                padding=True, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        reserve = 4 if self.mode == "sapt" else 0
        budget = self.max_new_tokens + reserve if reserve_generation else 0
        if inputs["input_ids"].shape[1] + budget > 4096:
            raise ValueError("Input and generation budget exceed LLaVA's 4096-token context")
        return inputs

    def _decode(self, ids, length):
        return self.tokenizer.batch_decode(ids[:, length:], skip_special_tokens=True)[0].strip(" ")

    @torch.inference_mode()
    def generate(self, question, image=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if self.reminder and self.reminder.token in question:
            raise ValueError("question contains the reserved SAPT token")
        if isinstance(image, Image.Image):
            image = image.copy()
        elif image is not None:
            with Image.open(image) as opened:
                image = opened.copy()
        prompt = self._format(question, image is not None)
        inputs = self._inputs(prompt, image)
        options = {}
        if self.reminder:
            self.reminder.reset(inputs)
            options = {"logits_processor": LogitsProcessorList([self.reminder]),
                       "stopping_criteria": StoppingCriteriaList([self.reminder])}
        ids = self.model.generate(**inputs, generation_config=self.generation_config, **options)
        answer = self._decode(ids, inputs["input_ids"].shape[1])
        triggered = self.reminder.triggered if self.reminder else False
        if triggered:
            # Keep legacy decode -> strip -> append reminder -> re-tokenize behavior.
            continuation = answer.rstrip() + " " + " ".join([self.reminder.token] * 4)
            second = self._inputs(prompt + " " + continuation.strip(), image)
            ids = self.model.generate(**second, generation_config=self.generation_config)
            answer = self._decode(ids, second["input_ids"].shape[1])
        return {"answer": answer, "reminder_triggered": triggered}
