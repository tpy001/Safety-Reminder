# SafetyReminder

This is the official repository of 

**The Safety Reminder: A Soft Prompt to Reactivate Delayed Safety Awareness in Vision-Language Models**

*Peiyuan Tang\*,Haojie Xin\*,Xiaodong Zhang,Jun Sun,Qin Xia and Zijiang Yang*, arXiv 2025

<a href='https://arxiv.org/pdf/2506.15734' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>


## Installation
```
conda create -n safeVLM python=3.10.14
conda activate safeVLM
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Datasets Download
#### 1. FigStep
download the images and question from https://github.com/ThuCCSLab/FigStep/tree/main/data 
You can git clone https://github.com/ThuCCSLab/FigStep.git and then move the images and questios to the data/FigStep folder.

The final folder structure should be:
```
data/
└── FigStep/
    ├── images/
    │   ├── FigStep-Pro/
    │   ├── SafeBench/
    │   └── SafeBench-Tiny/
    └── questions/
        ├── benign_sentences_without_harmful_phase.csv
        ├── SafeBench-Tiny.csv
        └── safebench.csv
```


## Scripts
#### 1. Generate the text and evaluate the safety
```
   python script/generate.py dataset=FigStep
   #  python script/generate.py dataset=VLSafe_harmful +dataset.sample=4
   #  python script/generate.py dataset=VLSafe_harmful +dataset.sample=4 debug=True
   #  python script/generate.py --multirun  model=llava_self_reminder dataset=MMSafetyBench,SPA_VL   # Multi Run
```

#### 2. Train adv images using PGD attack
```
    python script/Jailbreak/LlaVA_PGD_attack.py
```

#### 3. PromptTuning
Training
```
    python script/PromptTuning/train.py  --config_nam  llava_PT_training.yaml
```
Testing
```
    python script/generate.py model=llava_7b_PT dataset=VLSafe_harmful +dataset.sample=4 debug=True
```

#### 4. Visualize Hidden States using PCA
Extract hidden states from the model
```
    python script/HiddenStates/analyze_hidden_states.py --config-name hidden_states.yaml 
```

#### 5. Run SAPT
Training:
```
     python script/SAPT/train.py  debug=True
     # python script/SAPT/train.py model=Qwen2VL_SAPT debug=True

```
Testing:
```
    python script/generate.py model=llava_7b_SAPT dataset=VLSafe_harmful +dataset.sample=4 debug=True
```


#### 6. Test self reminder method
``` 
    python script/generate.py model=llava_self_reminder dataset=FigStep +dataset.sample=4 debug=True
```


#### 7. Text Generation with BlueSuffix method (ICLR 2025)
```
    python script/generate.py model=llava_7b_blue_suffix dataset=VLSafe_harmful +dataset.sample=4
```

#### 8. Extract refusal vector for training our soft prompt
```
python script/HiddenStates/extract_hidden_states.py --config-name hidden_states_train.yaml 
```