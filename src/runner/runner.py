from hydra import initialize, compose
from ..utils import set_seed,pretty_dict
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
logger.add(
    sink='train.log',  
    level='INFO',
    rotation='00:00', 
    compression='zip',        
    encoding='utf-8',  
    backtrace=True,
    enqueue=True,
    format= "{level} [{file}:{line}] {message}",
    mode='w'
) 
import os


class Runner:
    def __init__(self, config_dir,config_name):
        """
            Initializes the runner.
            Args:
                config_path: the path for the config file.
        """

        with initialize(config_path=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name)
            self.train_config = instantiate(cfg.train_config)
            self.model = instantiate(cfg.model)
            self.training_set = instantiate(cfg.training_set) 
            self.val_dataset = instantiate(cfg.val_dataset)
            self.evaluator = instantiate(cfg.evaluator) 

        logger.info(pretty_dict(cfg))

        # Step1: Set seed
        self.seed = self.train_config.get('seed') or 0
        set_seed(self.seed)


        # Step2: Build dataset
        self.train_batchsize = self.train_config.get('train_batchsize') if self.train_config.get('train_batchsize') is not None else 1
        self.val_batchsize = self.train_config.get('val_batchsize') if self.train_config.get('val_batchsize') is not None else 1

        # Step3: Build dataloader
        self.train_loader = torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.train_batchsize,
            shuffle=True,
            num_workers=2,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batchsize,
            shuffle=False,
            num_workers=2,
        )

        # Step4: Build optimizer
        self.optim_cfg = OmegaConf.to_container(cfg.train_config.get('optimizer'), resolve=True) 
        if self.optim_cfg is not None:
            optimzer_cls = self.optim_cfg.pop("type")
            if optimzer_cls == "AdamW":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.optim_cfg)
            else:
                raise ValueError("Unsupported optimizer type: %s" % optimzer_cls)

        logger.info("Training dataset size: %d" % len(self.training_set))
        logger.info("Val dataset size: %d" % len(self.val_dataset))
        
        # 4. build lr_schedule
        lr_schedule = cfg.train_config.get('lr_schedule')
        if lr_schedule is not None:
            raise NotImplementedError("lr_schedule is not implemented yet.")
     
        
       
    def test(self):
        # used for calculating the average time per token
        import time
        start_time = time.time()
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            responses = self.generate()
            # end_time = time.time()
            # generated_token_num = 0
            # for i in range(len(responses)):
            #     token_num = self.model.model.tokenizer(responses[i])['input_ids'][1:]
            #     generated_token_num += len(token_num)
            # print(f"生成的token数: {generated_token_num}")
            # elapsed_time = end_time - start_time
            # print(f"运行时间: {elapsed_time:.4f} 秒")
            # avg_time = generated_token_num / elapsed_time
            # print(f"平均每个token的生成时间: {avg_time:.4f} 秒")
            self.evaluation(responses)

    def train(self):
        self.model.train(True)  # Set model to training mode
        epoch = self.train_config.get('epoch')
        val_interval = self.train_config.get('val_interval')
        save_interval = self.train_config.get('save_interval')
        save_path = self.train_config.get('save_path')
        assert epoch is not None, "Please specify the number of epoch in the training_cfg"
        
        for i in range(epoch):
            with tqdm(total=len(self.train_loader), desc="Epoch %d" % (i)) as pbar:
                self.model.zero_grad()
                for iter, batch_inputs in enumerate(self.train_loader):
                    self.model.zero_grad()
                    loss = self.model(batch_inputs)
                    loss = loss.float()
                    loss.backward()
                    self.optimizer.step()
                    
                    pbar.update(1)  
                    pbar.set_postfix(loss=loss.item())  # 在进度条右侧显示损失值
                    
                    if iter % val_interval == 0 and iter > 0:
                        logger.info("Iter: %d, Loss: %f" % (i, loss.item()))
                        logger.info("Iter: %d\n" % (iter))
                        generated_text = []
                        for j in tqdm(range(0, len(self.val_dataset), self.val_batchsize), total=len(self.val_dataset) // self.val_batchsize, desc="Validating"):
                            torch.cuda.empty_cache()
                            self.model.eval()  # Set model to evaluation mode
                            with torch.no_grad():
                                outputs = self.model.generate(self.val_dataset[j:j+self.val_batchsize])
                                generated_text.extend(outputs)

                        for j in range(4):
                            logger.info("Question %d: %s" % (j,self.val_dataset[j]['question']))
                            logger.info("Reference %d: %s" % (j,self.val_dataset[j]['answer']))
                            logger.info("Generated text %d: %s" % (j,generated_text[j]))
                        torch.cuda.empty_cache()

                        with torch.no_grad():
                            safety_pred = self.evaluator.judge(generated_text,data={"question":self.val_dataset["question"]})

                        ASR = sum(1 for pred in safety_pred if pred == "false") / len(safety_pred)
                        logger.info(f"Attack success rate: {ASR:.4f}")
                        self.evaluator.destroy()
                        torch.cuda.empty_cache()
                        self.model.train(True)
                    
                    if iter % save_interval == 0:
                        self.model.save_model(os.path.join(save_path, 'prompt_embedding_iter_{}.pth'.format(i)))

    def generate(self):
        responses= []
        for i in tqdm(range(0, len(self.test_dataset), self.batchsize), desc="Generating responses"):
            inputs = self.test_dataset[i:i + self.batchsize]
            res = self.model.generate(inputs)
            responses.extend(res)

        return responses      

    def evaluation(self, responses):
        res = self.evaluator.evaluation(responses,self.test_dataset)
