"""SAPT reminder detection, using the model's standard generation loop."""

import numpy as np
import torch
from transformers import LogitsProcessor, StoppingCriteria


class SafetyReminder(LogitsProcessor, StoppingCriteria):
    def __init__(self, model, tokenizer, prompt_path, classifier_path):
        self.model = model
        state = torch.load(prompt_path, map_location="cpu", weights_only=True)
        weight = state.get("embedding.weight") if isinstance(state, dict) else None
        embedding = model.get_input_embeddings()
        if not isinstance(weight, torch.Tensor) or weight.shape != (4, embedding.embedding_dim):
            raise ValueError("SAPT checkpoint must contain embedding.weight with shape (4, hidden_size)")
        if not torch.isfinite(weight).all():
            raise ValueError("SAPT embedding contains non-finite values")
        self.weight = weight.float().to(embedding.weight.device)
        self.token = "<soft_prompt>"
        tokenizer.add_tokens([self.token])
        self.token_id = tokenizer.convert_tokens_to_ids(self.token)
        if self.token_id >= embedding.num_embeddings:
            raise ValueError("Soft prompt token is outside the model embedding vocabulary")
        with np.load(classifier_path, allow_pickle=False) as data:
            self.classifier = {key: torch.as_tensor(data[key].copy(), dtype=torch.float32,
                                                  device=embedding.weight.device)
                               for key in ("pca_components", "pca_mean", "clf_weights", "clf_bias")}
        dim = embedding.embedding_dim
        shapes = {"pca_mean": (dim,), "clf_weights": (1, 4), "clf_bias": (1,)}
        components = self.classifier["pca_components"]
        if components.ndim != 2 or components.shape[0] < 4 or components.shape[1] != dim:
            raise ValueError("Invalid PCA components shape")
        for key, shape in shapes.items():
            if tuple(self.classifier[key].shape) != shape:
                raise ValueError(f"Invalid {key} shape: expected {shape}")
        if not all(torch.isfinite(value).all() for value in self.classifier.values()):
            raise ValueError("Classifier contains non-finite values")
        self.hook = embedding.register_forward_hook(self._replace_embeddings)
        self.reset(None)

    def _replace_embeddings(self, module, args, output):
        mask = args[0] == self.token_id
        if not mask.any():
            return output
        if mask.sum().item() != 4:
            raise ValueError("Expected exactly four soft prompt tokens")
        # Replace only reserved positions; ordinary text and image embeddings remain unchanged.
        output = output.clone()
        output[mask] = self.weight.to(output.device, output.dtype)
        return output

    def reset(self, inputs):
        self.inputs = inputs
        self.triggered = False
        self.checks = []
        self.prompt_length = inputs["input_ids"].shape[1] if inputs is not None else 0

    def probability(self, input_ids):
        soft_ids = torch.full((1, 4), self.token_id, dtype=input_ids.dtype, device=input_ids.device)
        ids = torch.cat((input_ids, soft_ids), dim=1)
        extra = ids.shape[1] - self.inputs["attention_mask"].shape[1]
        mask = torch.cat((self.inputs["attention_mask"],
                          torch.ones((1, extra), dtype=self.inputs["attention_mask"].dtype,
                                     device=ids.device)), dim=1)
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        probe = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                 "use_cache": False, "output_hidden_states": True, "return_dict": True}
        if "pixel_values" in self.inputs:
            probe["pixel_values"] = self.inputs["pixel_values"]
        # The final soft token summarizes the query and partial response for safety detection.
        hidden = self.model(**probe).hidden_states[-1][:, -1]
        data = {key: value.to(hidden.device) for key, value in self.classifier.items()}
        # Frozen PCA + linear sigmoid head; this checkpoint predicts benign probability.
        projected = (hidden - data["pca_mean"]) @ data["pca_components"][:4].T
        return torch.sigmoid(projected @ data["clf_weights"].T + data["clf_bias"]).item()

    def __call__(self, input_ids, scores):
        # LogitsProcessor receives a tensor; StoppingCriteria receives a tuple/None.
        if isinstance(scores, torch.Tensor):
            count = input_ids.shape[1] - self.prompt_length
            if count % 16 == 15:
                probability = self.probability(input_ids)
                self.checks.append({"generated_tokens": count, "probability": probability})
                self.triggered = probability < 0.2
            return scores
        # Legacy code still appends one token after the unsafe check, then stops.
        return torch.tensor([self.triggered], device=input_ids.device)
