"""
Generate adversarial image for Llava-1.5-7B model to induce harmful content generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from datasets import load_dataset
from hydra.utils import instantiate
import logging
from transformers.image_utils import ImageInput,VideoInput,PILImageResampling,ChannelDimension,valid_images,validate_preprocess_arguments
from typing import Dict,Optional,Union,List 
from torch import TensorType
from transformers.models.qwen2_vl.image_processing_qwen2_vl import make_batched_images, make_batched_videos
from torchvision.transforms import Normalize


# Disable tokenizer parallelism to avoid issues with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.makedirs("logs", exist_ok=True)  # 自动创建 logs 文件夹

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/attack22.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

import PIL
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from transformers.image_processing_utils import BatchFeature
from torch.utils.data import DataLoader

def debug():
    import debugpy
    print("Waiting for debugger...")
    debugpy.listen(5678)
    debugpy.wait_for_client()


def distinct_n(corpus, n=1):
    """Compute Distinct-n for a list of generated sentences."""
    total_ngrams = 0
    unique_ngrams = set()
    for sentence in corpus:
        tokens = sentence.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngram_list = list(ngrams)
        total_ngrams += len(ngram_list)
        unique_ngrams.update(ngram_list)
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def set_preprocess_function(model, use_original_preprocessing: bool):
    """
    Dynamically set the preprocess function of the image processor inside the LlavaProcessor.

    Args:
        model: The model object containing the LlavaProcessor.
        use_original_preprocessing (bool): If True, use the original preprocessing function.
                                           If False, replace it with a custom one.

    Returns:
        The updated model with the appropriate preprocessing function set.
    """
    image_processor = model.processor.image_processor

    # Cache the original preprocess method if it hasn't been saved yet
    if not hasattr(model, "_original_preprocess"):
        model._original_preprocess = image_processor.preprocess

    if use_original_preprocessing:
        preprocess_func = model._original_preprocess
        model.processor.image_processor.do_rescale = True
    else:
        preprocess_func = custom_preprocess  # Your custom function must accept `self` as the first argument

        # Bind the custom function to the image_processor instance
        preprocess_func = preprocess_func.__get__(image_processor)
        model.processor.image_processor.do_rescale = False  # Disable rescaling for adversarial images

    # Replace the preprocess method
    image_processor.preprocess = preprocess_func

    return model


def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
    
        normalize = Normalize(mean=image_mean, std=image_std)

        # for i in range(images.shape[0]):
        #     images[i] = normalize(images[i])

        image = normalize(images[0]).unsqueeze(0)
        patches = image.repeat(self.temporal_patch_size, 1, 1, 1)
        resized_height,resized_width = patches.shape[2],patches.shape[3]
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

def custom_preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        batch_size = images.shape[0]
        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if images is not None:
            patches, image_grid_thw = _preprocess(
                self,
                images,
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
            )
            patches = patches.repeat(batch_size,1,1).flatten(0,1)
        new_img_grid = np.expand_dims(np.array(image_grid_thw), axis=0)
        image_grid_expanded = np.repeat(new_img_grid, repeats=batch_size, axis=0)
        data = {"pixel_values": patches, "image_grid_thw": image_grid_expanded}
        return BatchFeature(data=data, tensor_type=return_tensors)


# --- Define PGD Adversarial Attack Runner ---
class PGDJailbreakRunner:
    def __init__(self, model, train_config,train_loader,val_loader,test_loader,evaluator,val_dataset,*args, **kwargs):
        self.train_loader = train_loader 
        self.val_loader = val_loader 
        self.test_loader = test_loader  
        self.model = model
        self.batch_size = train_config.get("batch_size", 1)
        self.max_iter = train_config.get("iteration", 3000)
        self.step_size = train_config.get("step_size", 1/255)
        self.val_interval = train_config.get("val_interval", 300)

        self.work_dirs = train_config.get("work_dirs", "./work_dirs")
        self.output_img_dirs = os.path.join(self.work_dirs, "adv_imgs")
        os.makedirs(self.work_dirs, exist_ok=True)
        os.makedirs(self.output_img_dirs, exist_ok=True)

        self.clean_img_path = train_config["clean_image_path"]
        self.epsilon = train_config.get("epsilon", 254/255)

        self.evaluator = evaluator

        self.diversity_threshold = train_config.get("diversity_threshold", 0.)

        self.val_dataset = val_dataset

    def generate_random_noise(self, img, epsilon=None):
        if isinstance(img, Image.Image):
            size = [3, img.size[1], img.size[0]]
        else:
            size = img.shape

        if epsilon is None:
            noise = torch.rand(*size)  # [0,1]
        else:
            noise = torch.rand(*size) * 2 * epsilon - epsilon  # [-ε, +ε]
        return noise

    def train(self):
        self.model.train()

        # Modify the image processing function in Llava to preserve gradient flow.
        # By default, image preprocessing converts inputs to PIL images, which breaks gradient propagation.
        self.model = set_preprocess_function(self.model,use_original_preprocessing=False)

        num_iter = self.max_iter
        lr = self.step_size
        val_interval = self.val_interval

        img = Image.open(self.clean_img_path).convert("RGB")
        img_tensor = ToTensor()(img).to(self.model.device)

        adv_noise = self.generate_random_noise(img_tensor, self.epsilon).to(self.model.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        i = 0
        loss_curve = []
        best_ASR = 0.0
        while i < num_iter:
            for _, batch in enumerate(self.train_loader):
                inputs = {
                    "question": batch["goal"],
                    "chosen": batch["target"],
                }
                # Trick: Append a crafted prefix （1. To） to provoke harmful or policy-violating completions
                for j in range(len(inputs['chosen'])):
                    inputs['chosen'][j] += ": 1. To"

                adv_img = img_tensor + adv_noise
                adv_img = adv_img.clamp(0, 1).repeat(self.batch_size, 1, 1, 1)

                inputs["image"] = adv_img

                self.model.add_eos_token = False
                loss = self.model(inputs) # Delete the eos token at the end of the answer.
                loss.backward()

                # PGD update
                adv_noise.data = (adv_noise.data - lr * adv_noise.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
                adv_noise.data = (img_tensor + adv_noise.data).clamp(0, 1) - img_tensor

                adv_noise.grad.zero_()
                self.model.zero_grad()

                print("Iter: %d, Loss: %f" % (i, loss.item()))
                # if i % val_interval == 0:
                if i % val_interval == 0 and i > 0:
                    # plot the loss
                    loss_curve_path = os.path.join(self.work_dirs, "loss_curve.png")
                    plt.plot(loss_curve)
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.savefig(loss_curve_path)
                    plt.close()


                    logger.info(f"Iter: {i}, Loss: {loss.item():.4f}")
                    torch.cuda.empty_cache()

                    # Save adversarial image
                    adv_img_save = ToPILImage()(adv_img[0].detach().cpu())
                    adv_img_save.save(f"{self.work_dirs}/adv_noise_{i}.bmp")

                    # Evaluation on training dataset
                    with torch.no_grad():
                        generated_text = self.model.generate(inputs)
                    for j in range(len(generated_text)):
                        logger.info("Iter: %d, Generated Text %d: %s" % (i,j, generated_text[j]))


                    # Evaluation on val dataset
                    logger.info("Evaluating on val dataset...")
                    val_progress_bar = tqdm(self.val_loader, desc="Epoch [Current Epoch] Training")
                    output = []

                     
                    self.model = set_preprocess_function(self.model,use_original_preprocessing=True)
                    for _, val_batch in enumerate(val_progress_bar):
                        val_inputs = {
                            "question": val_batch["goal"],
                            "chosen": val_batch["target"],
                            "image": [ f"{self.work_dirs}/adv_noise_{i}.bmp" ] * len(val_batch["goal"])
                        }
                        generated_text = self.model.generate(val_inputs)
                        output.extend(generated_text)

                    # Evaluate the safety of the generated text
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        safety_pred = self.evaluator.judge(output,data={"question":self.val_dataset["goal"]})
                    ASR = sum(1 for pred in safety_pred if pred == "false") / len(safety_pred)
                    logger.info(f"Attack success rate: {ASR:.4f}")
                    self.evaluator.destroy()
                    torch.cuda.empty_cache()
                    self.model.train(True)

                    # Calculate the diversity of the generated text
                    distinct1 = distinct_n(output, 1)
                    distinct2 = distinct_n(output, 2)
                    distinct3 = distinct_n(output, 3)
                    distinct4 = distinct_n(output, 4)
                    distinct5 = distinct_n(output, 5)


                    logger.info(f"Distinct-1: {distinct1:.4f}, Distinct-2: {distinct2:.4f},Distinct-3: {distinct3:.4f},Distinct-4: {distinct4:.4f},Distinct-5: {distinct5:.4f}")
                    avg_diversity = (distinct1 + distinct2 + distinct3 + distinct4 + distinct5) / 5
                    logger.info(f"Iter: {i}, Avg. Diversity: {avg_diversity:.4f}")

                    # Calculate the attack success rate
                    unsafe_count = 0
                    for text in safety_pred:
                        if text.lower() == "false":
                            unsafe_count += 1
                    ASR = unsafe_count / len(safety_pred)
                    logger.info(f"Iter: {i}, Attack Success Rate: {ASR:.4f}")
                    if ASR >= best_ASR and avg_diversity >= self.diversity_threshold:
                        best_ASR = ASR
                        adv_img_save = ToPILImage()(adv_img[0].detach().cpu())
                        adv_img_save.save(f"{self.output_img_dirs}/adv_noise_best.bmp")

                    self.model = set_preprocess_function(self.model,use_original_preprocessing=False)

                
                i = i + 1
                loss_curve.append(loss.item())

        return adv_noise                
    
if __name__ == "__main__":
    # debug()
    # --- Load model configuration ---
    model_path = "weights/QWen2-VL-7B/"
    model_config = {
        "_target_": "src.model.VQAQwen2VL",
        "model_path": model_path,
        "torch_dtype": "bfloat16",
        "max_context_length": 1024,
        "device": "auto",
        "trainable": "frozen",
        "min_pixels": 12544, # 16 * 28 * 28  
        "max_pixels": 200704, # 256 * 28 * 28
        "generate_config": {
            "type": "GenerationConfig",
            "bos_token_id": 151643,
            "pad_token_id": 151643,
            "eos_token_id": [
                151645,
                151643
            ],
            "return_full_tex": False,
            "use_cache": True,
            "do_sample": False,
            # temperature: 0.6
            # top_p: 0.9
            "max_new_tokens": 256,
            "diversity_threshold": 0.1
        }
    }

    train_config = {
        "batch_size": 4,
        "work_dirs": "work_dirs/PGD_Qwen2VL",  # Directory to save adversarial images (default: BMP format)
        "epsilon": 64 / 255,  # Maximum allowable pixel perturbation (L∞ bound)
        "clean_image_path": "data/clean_imgs/clean_image1.jpg",  # Path to the clean input image
        "step_size": 1 / 255,  # Step size (learning rate) for PGD updates
        "iteration": 3000,  # Total number of PGD iterations
        "val_interval": 300,  # Interval (in steps) to evaluate and save the current adversarial image
    }

    evaluator_cfg = {
        "_target_": "src.evaluator.LLamaGuard3",
        "model_path": "weights/Llama3-Guard-8b/"
    }
    evaluator = instantiate(evaluator_cfg)

    # --- Load dataset ---
    train_dataset = load_dataset("csv", data_files="data/AdvBench/train.csv")['train']
    val_dataset = load_dataset("csv", data_files="data/AdvBench/val.csv")['train']
    test_dataset  = load_dataset("csv", data_files="data/AdvBench/test.csv")['train']

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 4),
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    # --- Instantiate model ---
    model = instantiate(model_config)

    runner = PGDJailbreakRunner(model,train_config,train_loader,val_loader,test_loader,evaluator,val_dataset=val_dataset)

    runner.train()
    runner.test()

