from .prompt_tuning import PromptTuningLlava, PromptTuningQwen2VL
import torch
import numpy as np
import torch.nn.functional as F

def classification(hidden_states: torch.Tensor, classfier_data: dict, top_m = 4,return_projected_data=False) -> torch.Tensor:
    device = hidden_states.device  
    
    pca_components = torch.tensor(classfier_data["pca_components"][:top_m], dtype=torch.float32, device=device)  # (m, D)
    pca_mean = torch.tensor(classfier_data["pca_mean"], dtype=torch.float32, device=device)                       # (D,)
    clf_weights = torch.tensor(classfier_data["clf_weights"], dtype=torch.float32, device=device)                 # (1, m)
    clf_bias = torch.tensor(classfier_data["clf_bias"], dtype=torch.float32, device=device)                       # (1,)

    centered_features = hidden_states - pca_mean           # shape: (N, D)
    projected = centered_features @ pca_components.T       # shape: (N, m)
    logits = projected @ clf_weights.T + clf_bias          # shape: (N, 1)
    probs_pos = torch.sigmoid(logits)                      # shape: (N, 1)

    if return_projected_data:
        return probs_pos, projected
    else:
        return probs_pos

class DRO_llava(PromptTuningLlava):
    PCA_DIM = 4

    def __init__(self, harmful_classfier_path, refusal_classfier_path, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.harmful_classfier_data = np.load(harmful_classfier_path)
        self.refusal_classfier_data = np.load(refusal_classfier_path)


    def forward(self,inputs,use_image=True,output_hidden_states=False):
        labels = inputs['safe']
        chosen = inputs.pop("chosen")
        batch_prompts,images = self.get_formatted_prompt(inputs,use_image)
        base_input = self.processor(images=images, text=batch_prompts, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **base_input,
                return_dict=True,
                output_hidden_states=True,
                use_original_forward=True,
            )
        base_hidden_states = outputs.hidden_states[-1][:,-1]

        formatted_prompt = [ self.add_soft_prompt(prompt) for prompt in batch_prompts]

        inputs = self.processor(images=images, text=formatted_prompt, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            soft_prompt_id = self.soft_prompt_id,
            soft_prompt_num = self.soft_prompt_num,
            soft_prompt_embedding = self.prompt_embedding
        )

        new_hidden_states = outputs.hidden_states[-1][:,-1]

        harmfulness_logits= classification(new_hidden_states,self.harmful_classfier_data,top_m=4).squeeze(-1)
        refusal_logits = classification(new_hidden_states,self.refusal_classfier_data,top_m=4).squeeze(-1)

        harmfulness_loss = F.binary_cross_entropy_with_logits(harmfulness_logits, labels.float().to(harmfulness_logits.device))
        refusal_loss = F.binary_cross_entropy_with_logits(refusal_logits, labels.float().to(refusal_logits.device))

        base_transformed = classification(base_hidden_states,self.harmful_classfier_data,top_m=4,return_projected_data=True)[1]
        new_transformed = classification(new_hidden_states,self.harmful_classfier_data,top_m=4,return_projected_data=True)[1]
        norm_loss = torch.mean((new_transformed - base_transformed)**2)
        
        return {
            "harmfulness_loss": harmfulness_loss,
            "refusal_loss": refusal_loss,
            "norm_loss": norm_loss,
            "total_loss":harmfulness_loss + refusal_loss * 1e-2 + norm_loss * 1e-3
        }
