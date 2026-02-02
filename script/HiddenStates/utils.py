from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, classification_report)

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