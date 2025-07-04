# SafeVLM
SafeVLM

## Installation
conda create -n safeVLM python=3.10.14
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

## Datasets Download
#### 1. FigStep
download the images and question from https://github.com/ThuCCSLab/FigStep/tree/main/data 


## Scripts
#### 1. Generate the text and evaluate the safety
```
    python script/generate.py --config_name llava_SPA_VL.yaml  --output_file  result.json
```

#### 2. Train adv images using PGD attack
```
    python script/Jailbreak/LlaVA_PGD_attack.py
```

#### 3. PromptTuning
Training
```
    python script/PromptTuning/train.py 
```
Testing
```
     python script/generate.py --config_name llava_PT.yaml  --output_file  result.json
```
