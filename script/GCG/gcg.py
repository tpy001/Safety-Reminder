
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlavaForConditionalGeneration
import hydra
from hydra.utils import instantiate
from hydra import initialize, compose
from src.utils import debug, set_seed
from datasets import load_dataset
import nanogcg

@hydra.main(version_base=None,config_path="../../configs/", config_name="test.yaml")  
def main(cfg) -> None:  
    is_debug = cfg.get("debug", False)
    if is_debug:
        debug()

    batch_size = cfg.get("batch_size", 1)
    # 1. Set seed
    seed = cfg.get("seed", 0)
    set_seed(seed)

    # 2. Load model
    vqa_model = instantiate(cfg.model)
    tokenizer = vqa_model.tokenizer
    model = vqa_model.model
   

    # 3. Load Dataset
    dataset = load_dataset("csv", data_files="script/GCG/harmful_behaviors.csv")["train"]

    config = GCGConfig(
        num_steps=500,
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING"
    )

    import json
    output_dir = "gcg_output"
    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

    max_samples = 100
    for i in range(max_samples):
        message = dataset[i]["goal"]
        target = dataset[i]["target"]

        result = nanogcg.run(model, tokenizer, message, target, config)

        result_dict = {
            "question": message,
            "best_string": result.best_string,
            "best_loss": result.best_loss,
            "losses": result.losses,
            "strings": result.strings
        }

        # 构造唯一的文件路径，比如 result_0.json, result_1.json, ...
        output_path = os.path.join(output_dir, f"result_{i}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        print(f"Saved result to {output_path}")

        # validation
        # jailbreak_output = vqa_model.generate(  
        #     {"question": message + result.best_string },
        #     use_image = False
        # )


if __name__ == "__main__":
    main()