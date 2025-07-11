from hydra import initialize, compose
from ..utils import set_seed,pretty_dict
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import hydra



class SAPT_Runner:
    def __init__(self, model,training_set,val_dataset,evaluator,train_config):
        """
            Initializes the runner.
            Args:
                config_path: the path for the config file.
        """

        self.train_config = train_config
        self.model = model
        self.training_set = training_set
        self.val_dataset = val_dataset
        self.evaluator = evaluator

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
        self.optim_cfg = OmegaConf.to_container(self.train_config.get('optimizer'), resolve=True) 
        if self.optim_cfg is not None:
            optimzer_cls = self.optim_cfg.pop("type")
            if optimzer_cls == "AdamW":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.optim_cfg)
            else:
                raise ValueError("Unsupported optimizer type: %s" % optimzer_cls)

        print("Training dataset size: %d" % len(self.training_set))
        print("Val dataset size: %d" % len(self.val_dataset))
        
        # 4. build lr_schedule
        lr_schedule = self.train_config.get('lr_schedule')
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
        val_interval = self.train_config.get('val_interval', 1)
        save_interval = self.train_config.get('save_interval', 1)

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_path = output_dir

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
                    pbar.set_postfix(
                        loss=loss.item(),
                    )
                    
                    # if iter % val_interval == 0 and iter > 0:
                    if iter % val_interval == 0:
                        # valiadation on training set
                        print("Iter: %d, Loss: %f" % (i, loss.item()))
                        generated_text = []
                        
                        torch.cuda.empty_cache()
                        self.model.eval()  # Set model to evaluation mode
                        with torch.no_grad():
                            for k in range(0, len(self.val_dataset), self.val_batchsize):
                                inputs = self.val_dataset[k:k + self.val_batchsize]
                                inputs.pop("chosen")
                                res = self.model.generate(inputs)
                                generated_text.extend(res)

                        for j in range(4):
                            print("Question %d: %s" % (j,batch_inputs['question'][j]))
                            print("chosen %d: %s" % (j,batch_inputs['chosen'][j]))
                            print("rejected %d: %s" % (j,batch_inputs['rejected'][j]))
                            print("Generated text %d: %s" % (j,generated_text[j]))
                        torch.cuda.empty_cache()

                        with torch.no_grad():
                            safety_pred = self.evaluator.judge(generated_text,data={"question":self.val_dataset["question"]})

                        ASR = sum(1 for pred in safety_pred if pred == "false") / len(safety_pred)
                        print(f"Attack success rate: {ASR:.4f}")
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

