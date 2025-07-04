import torch

class BaseModel(torch.nn.Module):
    def __init__(self,
                 model_path = "",
                 torch_dtype = 'float32',
                 device = None,
                 tokenizer =None,
                 *args,**kwargs):
        super().__init__()
        self.model_path= model_path
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None

        