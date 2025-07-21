import argparse
import yaml
import os

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from types import SimpleNamespace


from .utils import dict2namespace
from .diffpure_guided import GuidedDiffusion


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        #self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            # print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = (x_re + 1) * 0.5

        self.counter += 1

        return out
    
def debug():
    import debugpy
    print("Waiting for debugger...")
    debugpy.listen(5678)
    debugpy.wait_for_client()
   
def parse_args_and_config():
    args = SimpleNamespace(
        config="src/model/Image_Purifier/configs/diffpure.yml",
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
        log_dir="./image_pure_log"
    )
    # parse config file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    # # set random seed
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)

    # torch.backends.cudnn.benchmark = True
    
    
    return args, new_config


def diffpure(args, config,images):
    model = SDE_Adv_Model(args, config)
    model = model.eval().to(config.device)
    print('Model loaded!')
    image_size = config.model.image_size
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    batch_image = torch.stack([transform(img) for img in images]) 
    print('Processing image!')
    with torch.no_grad():
        output_images = model(batch_image)
    transform_back = transforms.ToPILImage()

    puried_images = []
    for idx, img_tensor in enumerate(output_images):
        img_pil = transform_back(img_tensor.cpu().clamp(0, 1))  # 转为 PIL 图像，确保在 [0,1] 范围
        puried_images.append(img_pil)
    return puried_images
  
    
def purify_image(images):
    if isinstance(images, str):
        images = [images]
    if isinstance(images[0], str):
        images = [Image.open(image_path) for image_path in images]

    args, config = parse_args_and_config()
    return diffpure(args, config,images)

    
