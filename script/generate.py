
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose
import torch
from src.utils import save_jailbreak_response,debug
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with specific config.")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../configs/",
        help="Path to the directory containing config files."
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default="llava_VLSafe.yaml",
        help="Name of the configuration file to use."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="./response.json",
        help="Path to save the output response."
    )

    return parser.parse_args()


def generate_text(model, questions,images=None,use_image=False,batch_size=1):
    """
    Generate text responses from a model in batches.

    Args:
        model: The model object that implements a `generate(inputs, use_image)` method.
        questions (str or List[str]): A single question or a list of questions.
        images (str or List[str], optional): A single image path or a list of image paths. 
            Required if `use_image` is True. Defaults to None.
        use_image (bool): Whether to use images as part of the input. Defaults to False.
        batch_size (int): The number of samples per batch. Defaults to 1.

    Returns:
        List[str]: A list of generated text outputs, one per input question.
    """
    if isinstance(questions,str):
        questions = [questions]
    if isinstance(images,str):
        images = [images]

    results = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating text"):
        if use_image:
            inputs = {
                "question": questions[i: i + batch_size],
                "image": images[i: i + batch_size]
            }
        else:
            inputs = {
                "question": questions[i: i + batch_size]
            }       
       
        generated_text = model.generate(inputs, use_image=use_image)
        results.extend(generated_text)
    return results

debug()
if __name__ == "__main__":
    from src.utils import set_seed

    args = parse_args()
    config_dir = args.config_dir
    config_name = args.config_name
    output_file = args.output_file


    with initialize(config_path=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
        model = instantiate(cfg.model.model)
        dataset = instantiate(cfg.dataset.dataset) 
        evaluator = instantiate(cfg.evaluator.evaluator) 

    seed = cfg.get("seed", 0)
    batch_size = cfg.get("batch_size", 1)
    set_seed(seed)

    test_data = dataset[:]
    if isinstance(test_data['image'][0], str):
        image_path = test_data['image']
    else:
        image_path = test_data['image_path']
    output_text = generate_text(
        model,
        test_data['question'],
        test_data['image'] ,
        use_image=True,
        batch_size=batch_size
    )

    del model
    torch.cuda.empty_cache()  
    safe_pred = evaluator.judge(pred=output_text, data=test_data['ori_question'])

    data = {
        "question": test_data['question'],
        "ori_question": test_data['ori_question'],
        "answer": test_data['answer'],
        "category": test_data['category'],
    }
    save_jailbreak_response(output_text, data, safe_pred, image_path = image_path,output_file_path=output_file)





