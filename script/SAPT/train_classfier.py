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
from sklearn.linear_model import LogisticRegression

from src.utils import debug, set_seed
import hydra 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, classification_report)

def evaluate_with_saved_classifier(hidden_states: np.ndarray,
                                   true_labels: np.ndarray,
                                   npz_path: str,
                                   top_m: int = 4,
                                   pos_label: int = 0,
                                   plot: bool = True,
                                save_path: str = "roc_curve.png"):
    data = np.load(npz_path)
    pca_components = data["pca_components"][:top_m]  # (m, D)
    pca_mean = data["pca_mean"]                      # (D,)
    clf_weights = data["clf_weights"]                # (1, m)
    clf_bias = data["clf_bias"]                      # (1,)

    # PCA 中心化并投影
    centered_features = hidden_states - pca_mean
    projected = centered_features @ pca_components.T  # shape: (N, m)

    logits = projected @ clf_weights.T + clf_bias     # shape: (N, 1)
    probs_pos  = 1 / (1 + np.exp(-logits))                 # sigmoid

    if pos_label == 0:
        probs = 1 - probs_pos  # 负类的概率
    else:
        probs = probs_pos
    
    preds = (probs > 0.5).astype(int).flatten()

    for i in range(len(true_labels)):
        true_labels[i] = not true_labels[i]

    # 计算指标
    results = {
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds, pos_label=pos_label),
        "recall": recall_score(true_labels, preds, pos_label=pos_label),
        "f1": f1_score(true_labels, preds, pos_label=pos_label),
        "auc": roc_auc_score(true_labels, probs)
    }

    print("Classification Report:\n", classification_report(true_labels, preds))
    print("Evaluation Results:", results)

    if plot:
        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(true_labels, probs)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {results['auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        # 绘制 PR 曲线
        precision, recall, _ = precision_recall_curve(true_labels, probs, pos_label=pos_label)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        plt.tight_layout()
        plt.show()
        plt.savefig(save_path)  # 保存图像


    return results


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

    def fit_and_save_classifier(self, m=4, save_name="pca_classifier.npz"):
        """
        在前 m 个 PCA 主成分上训练逻辑回归分类器并保存模型参数和主成分向量。

        Args:
            m (int): 使用前 m 个主成分进行训练
            save_name (str): 保存文件的名称（npz格式）
        """
        assert m <= self.n_components, f"m={m} 不能超过设置的 n_components={self.n_components}"

        X = self.pca.transform(self.features)[:, :m]  # shape: (N, m)
        y = self.labels

        clf = LogisticRegression()
        clf.fit(X, y)

        # 保存内容：
        # - pca_components: shape (m, D)
        # - clf.coef_: shape (1, m)
        # - clf.intercept_: shape (1,)
        save_path = os.path.join(self.output_dir, save_name)
        np.savez(save_path,
            pca_components=self.pca.components_[:m],
            pca_mean=self.pca.mean_,                 # ✅ 加上这个
            clf_weights=clf.coef_,
            clf_bias=clf.intercept_)

        print(f"[Saved] PCA components and classifier weights saved to: {save_path}")

    def fit_classifier_and_plot(self):
        """
        在前两个 PCA 分量上训练逻辑回归分类器并可视化决策边界
        """
        X = self.converted_data[:, :2]  # 只用前两个主成分
        y = self.labels

        clf = LogisticRegression()
        clf.fit(X, y)

        # 创建网格用于绘制决策边界
        h = .02  # 网格间距
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)

        # 再绘制原始点
        marker_map = {0: 'X', 1: 'o'}
        unique_datasets = sorted(list(set(self.dataset_names)))
        colors = matplotlib.colormaps['tab10'].colors
        color_map = {d: colors[i % len(colors)] for i, d in enumerate(unique_datasets)}

        combos = sorted(list(set(zip(self.labels, self.dataset_names))))
        for s, d in combos:
            mask = (self.labels == s) & (self.dataset_names == d)
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                color=color_map[d],
                marker=marker_map[s],
                alpha=0.8,
                edgecolors='none',
                s=50
            )

        # 添加图例与标签
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

        ax.set_xlabel('First Principal Component', fontsize=14)
        ax.set_ylabel('Second Principal Component', fontsize=14)
        ax.set_title('PCA with Logistic Regression Decision Boundary', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'PCA_Logistic_DecisionBoundary.png'))
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
    visualizer.fit_classifier_and_plot()

    #####  ######   ######   ######   ######   ######   ###### 
    use_image = True
    ###### Important ！  ######   ######   ######   ###### 
    val_features, val_labels = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(val_dataset), batch_size)):
            batch = val_dataset[i:i + batch_size]

            output = model(batch, output_hidden_states=True,use_answer=use_answer,use_image=use_image)

            # 只取指定层和token
            hs_tuple = output if isinstance(output, tuple) else output.hidden_states
            selected_layer = hs_tuple[cfg.select_layer]  # shape: (B, T, D)
            token_vec = selected_layer[:, cfg.select_token_index, :]  # shape: (B, D)

            val_features.append( token_vec.float().cpu().numpy())
            safe_labels = batch["safe"]
            if isinstance(safe_labels, torch.Tensor):
                safe_labels = safe_labels.cpu().tolist()
            val_labels.extend(safe_labels)

    val_features = np.vstack(val_features)
    visualizer.fit_and_save_classifier(m=4, save_name="pca_classifier.npz")


    evaluate_with_saved_classifier(features,
                                   labels,
                                   os.path.join(visualizer.output_dir,  "pca_classifier.npz")
                                )
    
    evaluate_with_saved_classifier(val_features,
                                   val_labels,
                                   os.path.join(visualizer.output_dir,  "pca_classifier.npz")
                                )



if __name__ == '__main__':
    debug()
    main()
