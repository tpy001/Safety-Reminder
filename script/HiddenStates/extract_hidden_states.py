import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from hydra.utils import instantiate
from hydra import initialize, compose
import matplotlib
from matplotlib.lines import Line2D 

from src.utils import debug, set_seed
import hydra 

def top_tokens_from_hidden(hidden_state, tokenizer, lm_head, topk=10):
    """
    给定隐藏状态向量，返回通过 lm_head 映射后的 top-k 对应 token 字符串。

    Args:
        hidden_state (np.ndarray or torch.Tensor): 单个隐藏状态向量（形状: [hidden_dim]）
        tokenizer: HuggingFace 的 tokenizer 实例
        lm_head: 通常为 LLM 中的 lm_head，例如 model.lm_head
        topk (int): 返回前多少个 token（默认10）

    Returns:
        List[str]: top-k 对应的 token 字符串
    """
    if not isinstance(hidden_state, torch.Tensor):
        hidden_state = torch.tensor(hidden_state)

    # 确保在正确的设备和 dtype 上
    hidden_state = hidden_state.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)

    # 通过 lm_head 得到 logits
    logits = lm_head(hidden_state)  # shape: [vocab_size]

    # 取 top-k
    topk_values, topk_indices = torch.topk(logits, k=topk)

    # 转换为 token 字符串
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())

    return topk_tokens

debug()
class DataVisualizer:
    def __init__(self, features, labels, dataset_names,n_components=2):
        self.labels = np.array(labels)    
        self.dataset_names = np.array(dataset_names)
        self.features = features
        self.n_components = n_components
        self._run_pca()

        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    def _run_pca(self):
        self.pca = PCA(n_components=self.n_components)
        self.converted_data = self.pca.fit_transform(self.features)
        self.explained_variance = self.pca.explained_variance_ratio_


    def visualize_data(self):
        """
        Visualizes the data using a clear strategy:
        - COLOR represents the dataset.
        - MARKER SHAPE represents the safety status.
        """
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. Define markers
        marker_map = {
            0: 'X',  # Circle for "Safe"
            1: 'o'   # 'X' for "Harmful"
        }

        # 2. Define colors
        unique_datasets = sorted(list(set(self.dataset_names)))
        colors = matplotlib.colormaps['tab10'].colors
        color_map = {d: colors[i % len(colors)] for i, d in enumerate(unique_datasets)}

        # 3. Plot the data points
        combos = sorted(list(set(zip(self.labels, self.dataset_names))))
        for s, d in combos:
            mask = (self.labels == s) & (self.dataset_names == d)
            ax.scatter(
                self.converted_data[mask, 0],
                self.converted_data[mask, 1],
                color=color_map[d],
                marker=marker_map[s],
                alpha=0.8,
                edgecolors='none',
                s=50  # CHANGED: Increased point size from 30 to 50
            )

        # 4. Create legends
        marker_legend_elements = [
            Line2D([0], [0], marker='o', color='gray', label='Safe', linestyle='None', markersize=10),
            Line2D([0], [0], marker='X', color='gray', label='Harmful', linestyle='None', markersize=10)
        ]
        legend1 = ax.legend(handles=marker_legend_elements, loc='upper right', title="Safety Status", fontsize='large')
        ax.add_artist(legend1)

        color_legend_elements = [
            Line2D([0], [0], marker='o', color=c, label=d, linestyle='None', markersize=10)
            for d, c in color_map.items()
        ]
        ax.legend(handles=color_legend_elements, loc='lower right', title="Datasets", ncol=2, fontsize='large')

        # 5. Finalize plot with larger fonts
        # CHANGED: Added fontsize to labels and title
        ax.set_xlabel('First Principal Component', fontsize=14)
        ax.set_ylabel('Second Principal Component', fontsize=14)
        ax.set_title('PCA of Hidden States by Dataset & Safety', fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'VisualData_PCA_Refactored.png'))
        plt.show()


    def visualize_cumulative_var(self):
        plt.clf()
        cum_var = np.cumsum(self.explained_variance)
        plt.figure(figsize=(10, 6)) # Added figure size for better layout
        
        # CHANGED: Increased line marker size
        plt.plot(range(1, self.n_components + 1), cum_var, marker='o', markersize=8)
        
        # CHANGED: Added fontsize to labels and title
        plt.xlabel('Number of Components', fontsize=14)
        plt.ylabel('Cumulative Explained Variance', fontsize=14)
        plt.title('Explained Variance by PCA', fontsize=16)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'CumulativeVar_PCA.png'))
        plt.show()

@hydra.main(version_base=None,config_path="../../configs/", config_name="test.yaml")  
def main(cfg) -> None:  
    is_debug = cfg.get("debug", False)
    if is_debug:
        debug()

    batch_size = cfg.get("batch_size", 1)
    n_components = cfg.get("n_components", 20)
    use_answer  = cfg.get("use_answer", False) 
    # 1. Set seed
    seed = cfg.get("seed", 0)
    set_seed(seed)

    # 2. Load model
    # model = instantiate(cfg.model)
    model = instantiate(cfg.model)


    # 3. Load dataset
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)


    # 4. Load evaluator
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(train_dataset), batch_size)):
            batch = train_dataset[i:i + batch_size]
    
            output = model(batch, output_hidden_states=True,use_answer=use_answer,use_image=False)

            # 只取指定层和token
            hs_tuple = output if isinstance(output, tuple) else output.hidden_states
            selected_layer = hs_tuple[cfg.select_layer]  # shape: (B, T, D)
            token_vec = selected_layer[:, cfg.select_token_index, :]  # shape: (B, D)

            features.append( token_vec.float().cpu().numpy())
            safe_labels = batch["safe"]
            if isinstance(safe_labels, torch.Tensor):
                safe_labels = safe_labels.cpu().tolist()
            labels.extend(safe_labels)
    
    features = np.vstack(features)
    visualizer = DataVisualizer(features, train_dataset["safe"], train_dataset["source_dataset"], n_components=n_components)
    visualizer.visualize_data()
    visualizer.visualize_cumulative_var()
    top_tokens_from_hidden(visualizer.pca.components_[0],model.tokenizer,model.get_language_model().lm_head,topk=20)


if __name__ == '__main__':
    main()
