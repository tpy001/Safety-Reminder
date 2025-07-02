from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch

import torch.nn.functional as F
from torchvision import transforms

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import(
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)

from transformers.utils import TensorType, is_vision_available, logging

def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample= None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        size_ = (336,336)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if isinstance(images, PIL.Image.Image):
             images = torch.tensor(np.array(images).transpose(2, 0, 1)).float() / 255.0

        if images.ndim == 3:
            adv_image = images.unsqueeze(0)
        else:
            adv_image = images
        
        if do_resize:
            adv_image = F.interpolate(adv_image, size=size_, mode='bicubic', align_corners=False)

        if do_rescale:
            if torch.max(adv_image) <= 1:
                raise ValueError("Input image has been normalized to be in the range [0, 1]")
            else:
                adv_image = adv_image * rescale_factor

        if do_normalize:
            adv_image = transforms.Normalize(mean=image_mean, std=image_std)(adv_image)

        data = {"pixel_values": adv_image}
        return BatchFeature(data=data, tensor_type=return_tensors)


