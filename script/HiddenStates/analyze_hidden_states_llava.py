
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets
import json


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import debug, set_seed
from src.model.llava import VQALlaVA
from src.model.SAPT import SAPTLlava
from src.model.Qwen2VL import VQAQwen2VL    
from src.model.SAPT_Qwen import SAPTQwen2VL 
from script.HiddenStates.utils import top_tokens_from_hidden,evaluate_with_saved_classifier
import hydra 
import re

def truncate_to_full_sentence(text, ratio):
    """
    在不超过 ratio*len(text) 的前提下，截断到最后一个完整的英文句子。
    截断点后，下一个词必须以大写字母开头，且不是换行符。
    忽略编号形式（如 "1. Dog"）的误判。
    """
    max_chars = int(len(text) * ratio)
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # 改进正则：排除出现在数字后的句号，如 "1." 或 "2."
    pattern = r"""
        (?<!\d)          # 前面不是数字
        [.!?]            # 匹配句号、问号、感叹号
        [ \t]*           # 可有空格或制表符（不含换行）
        (?=[A-Z])        # 后面紧跟大写字母
    """

    matches = list(re.finditer(pattern, truncated, re.VERBOSE))
    if matches:
        end = matches[-1].end()
        return truncated[:end].strip()

    # fallback：找最后的句号/问号/感叹号或换行符
    fallback_pattern = r"(?<!\d)[.!?](\s|\n)*|\n+"
    fallback_matches = list(re.finditer(fallback_pattern, truncated))
    if fallback_matches:
        end = fallback_matches[-1].end()
        return truncated[:end].strip()

    return truncated.strip()
    
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
        fig, ax = plt.subplots(figsize=(8, 6))
        # Marker & color map
        style_map = {
            0: {'label': 'Safe', 'marker': 'o', 'color': '#1f77b4'},    # 蓝色圆圈
            1: {'label': 'Harmful', 'marker': 'X', 'color': '#d62728'}  # 红色 X
        }

        # 绘图
        for s in sorted(set(self.labels)):
            mask = self.labels == s
            ax.scatter(
                self.converted_data[mask, 0],
                self.converted_data[mask, 1],
                color=style_map[s]['color'],
                marker=style_map[s]['marker'],
                alpha=0.85,
                s=60,
                edgecolors='black' if style_map[s]['marker'] == 'o' else 'none',  # 给圆圈加边界
                label=style_map[s]['label']
            )

        # 图例（直接从 label 获取）
        ax.legend(title="Safety Label", fontsize='large', title_fontsize='large')

        # 坐标轴和标题
        ax.set_xlabel('First Principal Component', fontsize=14)
        ax.set_ylabel('Second Principal Component', fontsize=14)
        ax.set_title('PCA of Hidden States (Safe vs Harmful)', fontsize=16)

        # 美化
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'VisualData_PCA_SingleDataset.png'))
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
        X = self.converted_data[:, :2]
        y = self.labels  # 0: harmful, 1: safe

        # 1. 拟合逻辑回归分类器
        clf = LogisticRegression()
        clf.fit(X, y)

        # 2. 创建网格并预测
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # 3. 绘图
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.2)

        # 4. 点图样式：蓝色Safe，红色Harmful
        style_map = {
            0: {'label': 'Harmful', 'marker': 'X', 'color': '#d62728'},
            1: {'label': 'Safe',    'marker': 'o', 'color': '#1f77b4'}
        }

        for label in sorted(np.unique(y)):
            mask = (y == label)
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=style_map[label]['color'],
                marker=style_map[label]['marker'],
                label=style_map[label]['label'],
                s=60,
                alpha=0.85,
                edgecolors='black' if style_map[label]['marker'] == 'o' else 'none'
            )

        # 5. 图例与标签
        ax.legend(title='Safety Label', fontsize='large', title_fontsize='large')
        ax.set_xlabel('First Principal Component', fontsize=14)
        ax.set_ylabel('Second Principal Component', fontsize=14)
        ax.set_title('PCA with Logistic Regression Decision Boundary', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'PCA_Logistic_DecisionBoundary.png'))
        plt.show()


@hydra.main(version_base=None,config_path="./configs", config_name="config.yaml")  
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
    model = instantiate(cfg.model)
    tokenizer = model.tokenizer

    # 3. Load dataset from saved json files
    if type(model) in [VQALlaVA,SAPTLlava]:
        harmful_data_path = "./script/HiddenStates/data/llava_harmful/genereated_response.json"
        benign_data_path = "./script/HiddenStates/data/llava_benign/genereated_response.json"
    elif type(model)  in [VQAQwen2VL,SAPTQwen2VL]:
        harmful_data_path = "./script/HiddenStates/data/Qwen2-harmful/genereated_response.json"
        benign_data_path = "./script/HiddenStates/data/Qwen2-benign/genereated_response.json"
    else:
        print("Unsupport model type !")

    with open(harmful_data_path, "r") as f:
        harm_data_json = json.load(f)

    with open(benign_data_path, "r") as f:
        safe_data_json = json.load(f)

    harm_data = {
        "question": [],
        "chosen": [],
        "safe": [],
        "image": [],
        "source_dataset": ["unknown"] * len(harm_data_json) 
    }

    safe_data = {
        "question": [],
        "chosen": [],
        "safe": [],
        "image": [],
        "source_dataset": ["unknown"] * len(safe_data_json) 
    }

    max_new_token = 0.5 # TODO: Modify this parameters to 0 if you want to visualize the hidden state at the begining of text generation.
    tokenizer = model.tokenizer

    for i in range(len(harm_data_json)):
        harm_data["question"].append(harm_data_json[i]["question"])
        # text = harm_data_json[i]["response"]
        # response_text = truncate_to_full_sentence(text, max_new_token)
        # response_tokens = tokenizer.tokenize(response_text)
        # response_text = tokenizer.convert_tokens_to_string(response_tokens)

        tokens = tokenizer.tokenize( harm_data_json[i]["response"])
        length = int(max_new_token* len(tokens))
        response_tokens =tokens[:length]
        response_text = tokenizer.convert_tokens_to_string(response_tokens)

        harm_data["chosen"].append(response_text)
        harm_data["safe"].append("false")
        # harm_data["safe"].append( harm_data_json[i]["safe"])
        harm_data["image"].append(harm_data_json[i]["image_path"])

    for i in range(len(safe_data_json)):
        safe_data["question"].append(safe_data_json[i]["question"])

        # text = safe_data_json[i]["response"]
        # response_text = truncate_to_full_sentence(text, max_new_token)

        # response_tokens = tokenizer.tokenize(response_text)
        # response_text = tokenizer.convert_tokens_to_string(response_tokens)


        tokens = tokenizer.tokenize( safe_data_json[i]["response"])
        length = int(max_new_token* len(tokens))
        response_tokens =tokens[:length]
        response_text = tokenizer.convert_tokens_to_string(response_tokens)

        safe_data["chosen"].append(response_text)
        safe_data["safe"].append("true")
        # safe_data["safe"].append( safe_data_json[i]["safe"])
        safe_data["image"].append(safe_data_json[i]["image_path"])

 
    harm_dataset = Dataset.from_dict(harm_data)
    benign_dataset = Dataset.from_dict(safe_data)

    train_dataset = concatenate_datasets([harm_dataset, benign_dataset])

    # 4. Load evaluator
    model.eval()
    features, labels = [], []

    if max_new_token > 0:
        use_answer = True

    with torch.no_grad():
        for i in tqdm(range(0, len(train_dataset), batch_size)):
            batch = train_dataset[i:i + batch_size]

            if type(model) in [VQALlaVA,VQAQwen2VL]:
                output = model(batch, output_hidden_states=True,use_answer=use_answer)
            elif type(model)  in [SAPTLlava,SAPTQwen2VL]:
                output = model.forward_new(batch, output_hidden_states=True)
            else:
                print("Warning: unsupport model type!")
            

            # 只取指定层和token
            hs_tuple = output if isinstance(output, tuple) else output.hidden_states
            hs_tuple = hs_tuple[1]
            selected_layer = hs_tuple[cfg.select_layer]  # shape: (B, T, D)
            token_vec = selected_layer[:, cfg.select_token_index, :]  # shape: (B, D)

            features.append(token_vec.float().cpu().numpy())
            safe_labels = batch["safe"]
            if isinstance(safe_labels, torch.Tensor):
                safe_labels = safe_labels.cpu().tolist()
            labels.extend(safe_labels)

    labels = [l.lower() == 'true' for l in labels]

    features = np.vstack(features)
    not_nan_mask = ~np.isnan(features).any(axis=1)
    features = features[not_nan_mask]
    labels = [label for label, keep in zip(labels, not_nan_mask) if keep]
    source_dataset = [source for source, keep in zip(train_dataset["source_dataset"], not_nan_mask) if keep]

    # visualizer = DataVisualizer(features, train_dataset["safe"], train_dataset["source_dataset"], n_components=n_components)
    visualizer = DataVisualizer(features, labels, source_dataset, n_components=n_components)
    np.save(
        os.path.join(visualizer.output_dir, "hidden_states.npy"),
        features
    )
    visualizer.visualize_data()
    visualizer.visualize_cumulative_var()
    try:
        lm_head = model.get_language_model().lm_head
    except:
        lm_head = model.model.lm_head

    top_tokens_from_hidden(visualizer.pca.components_[0],model.tokenizer,lm_head,topk=20)
    visualizer.fit_classifier_and_plot()

    visualizer.fit_and_save_classifier(m=2, save_name="pca_classifier.npz")
    evaluate_with_saved_classifier(features,
                                   labels,
                                   os.path.join(visualizer.output_dir,  "pca_classifier.npz")
                                )


if __name__ == '__main__':
    main()
