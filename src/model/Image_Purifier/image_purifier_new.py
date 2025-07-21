import os
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from types import SimpleNamespace
from .diffpure_guided import GuidedDiffusion
from .utils import str2bool, dict2namespace

class SDE_Adv_Model(torch.nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.counter.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
        out = (x_re + 1) * 0.5
        self.counter += 1
        return out

from typing import Union, List
def purify_image(
    input_image: Union[str, Image.Image, List[Image.Image], List[str], torch.Tensor],
    config_path="src/model/Image_Purifier/configs/diffpure.yml",
    log_dir="./image_pure_log"
) -> Union[Image.Image, List[Image.Image]]:
    """
    输入图像或图像列表，返回 purified 图像或列表。
    支持输入：路径 / PIL.Image / Tensor / 批处理
    """

    args = SimpleNamespace(
        config=config_path,
        data_seed=0,
        seed=1234,
        exp='exp',
        verbose='info',
        image_folder='images',
        ni=True,
        sample_step=1,
        t=400,
        t_delta=15,
        rand_t=False,
        diffusion_type='ddpm',
        score_type='guided_diffusion',
        eot_iter=20,
        use_bm=False,
        sigma2=1e-3,
        lambda_ld=1e-2,
        eta=5.0,
        step_size=1e-3,
        domain='imagenet',
        classifier_name='Eyeglasses',
        partition='val',
        adv_batch_size=64,
        attack_type='square',
        lp_norm='Linf',
        attack_version='custom',
        num_sub=1000,
        adv_eps=0.07,
        log_dir=log_dir
    )

    with open(config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    config = dict2namespace(raw_cfg)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SDE_Adv_Model(args, config).eval().to(config.device)
    print("Model loaded.")

    image_size = config.model.image_size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # 标准化输入为 batch tensor
    if isinstance(input_image, str):
        input_image = [Image.open(input_image).convert("RGB")]
    elif isinstance(input_image, Image.Image):
        input_image = [input_image]
    elif isinstance(input_image, list) and all(isinstance(i, str) for i in input_image):
        input_image = [Image.open(p).convert("RGB") for p in input_image]
    elif isinstance(input_image, torch.Tensor):
        image_tensor = input_image.to(config.device)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
    elif isinstance(input_image, list) and all(isinstance(i, Image.Image) for i in input_image):
        pass
    else:
        raise ValueError("Unsupported input_image format")

    # 如果不是tensor，转成 batch tensor
    if not isinstance(input_image, torch.Tensor):
        image_tensor = torch.stack([transform(img) for img in input_image], dim=0).to(config.device)

    # Purify
    with torch.no_grad():
        purified_batch = model(image_tensor)

    purified_batch = torch.clamp(purified_batch, 0, 1).cpu()
    to_pil = transforms.ToPILImage()
    result = [to_pil(img.squeeze(0)) if img.ndim == 3 else to_pil(img) for img in purified_batch]

    return result[0] if len(result) == 1 else result
