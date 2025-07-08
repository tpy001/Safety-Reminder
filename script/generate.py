
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import hydra
from omegaconf import OmegaConf
import torch
from src.utils import save_jailbreak_response,debug
from tqdm import tqdm
from hydra.utils import instantiate

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

from src.utils import set_seed

@hydra.main(version_base=None,config_path="../configs/", config_name="test.yaml")  
def main(cfg) -> None:  
    is_debug = cfg.get("debug", False)
    if is_debug:
        debug()

    batch_size = cfg.get("batch_size", 1)
    # 1. Set seed
    seed = cfg.get("seed", 0)
    set_seed(seed)

    # 2. Load model
    model = instantiate(cfg.model)

    # 3. Load dataset
    dataset = instantiate(cfg.dataset)

    # 4. Load evaluator
    evaluator = instantiate(cfg.evaluator)

    # 5. Generate text
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
    safe_pred = evaluator.judge(answers=output_text, data=test_data['ori_question'])

    data = {
        "question": test_data['question'],
        "ori_question": test_data['ori_question'],
        "chosen": test_data['chosen'],
        "category": test_data['category'],
    }

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_jailbreak_response(output_text, data, safe_pred, image_path = image_path,output_file_path=os.path.join(output_dir,"genereated_response.json"))



if __name__ == "__main__":
    main()
   





