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
├── data/
    ├── FigStep/
        ├── images/
        │   ├── FigStep-Pro/
        │   ├── SafeBench/
        │   └── SafeBench-Tiny/
        └── questions/
            ├── benign_sentences_without_harmful_phase.csv
            ├── SafeBench-Tiny.csv
            └── safebench.csv
```

#### 2. MMSafetyBench
Please see https://github.com/isXinLiu/MM-SafetyBench

The final folder structure should be:

```
├── data    
    ├── MMSafetyBench/
        ├── processed_questions
            ├── 01-Illegal_Activitiy.json
            ├── 02-HateSpeech.json
            ├── 03-Malware_Generation.json
            └── ... # {scenario}.json
        ├── imgs
            ├── 01-Illegal_Activitiy
                ├── SD
                    ├── 0.jpg
                    ├── 1.jpg
                    ├── 2.jpg
                    └── ... # {question_id}.jpg
                ├── SD_TYPO
                    ├── 0.jpg
                    ├── 1.jpg
                    ├── 2.jpg
                    └── ... # {question_id}.jpg
                ├── TYPO
                    ├── 0.jpg
                    ├── 1.jpg
                    ├── 2.jpg
                    └── ... # {question_id}.jpg
            ├── 02-HateSpeech
            ├── 03-Malware_Generation
            └── ...
```

#### 3. VLSafe
Please download from our google drive.


## Judge Model Download
Download the Llama-Guard3 from the 

## Scripts
#### 1. Generate the text and evaluate the safety
```
   python script/generate.py dataset=FigStep
   #  python script/generate.py dataset=VLSafe_harmful +dataset.sample=4 # Sample 4 data from the dataset
   #  python script/generate.py dataset=VLSafe_harmful +dataset.sample=4 debug=True # Debug the code 
   #  python script/generate.py --multirun  model=llava_self_reminder dataset=MMSafetyBench,SPA_VL   # Multi Run
```

#### 2. Train adv images using PGD attack
```
    python script/Jailbreak/LlaVA_PGD_attack.py
    # python script/Jailbreak/Qwen2VL_PGD_attack.py # For Qwen2VL model
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


#### 4. Run SAPT
Training:
```
     python script/SAPT/train.py  
     # python script/SAPT/train.py model=Qwen2VL_SAPT debug=True # For Qwen2VL model and debug the code 

```
Testing:
```
    python script/generate.py model=llava_7b_SAPT dataset=FigStep +dataset.sample=4 debug=True
    # python script/generate.py model=Qwen2VL_SAPT dataset=FigStep +dataset.sample=4 debug=True # Qwen2VL model
```

#### 5. Test self reminder method (Nature)
``` 
    python script/generate.py model=llava_self_reminder dataset=FigStep +dataset.sample=4 debug=True
```

#### 6. Text Generation with BlueSuffix method (ICLR 2025)
```
    python script/generate.py model=llava_7b_blue_suffix dataset=VLSafe_harmful +dataset.sample=4
```

#### 7. DRO model (ICML 2024)
Training:
```
# Step1: Training a refusal and a harmfulness classifier
python script/DRO/train_cls_DRO.py --config-name classfier_train model=llava_7b

# Step2: Direct Representation Optimization(DRO) for safety prompt
python script/DRO/train.py --config-name train_DRO.yaml model=llava_7b_DRO
```
Testing:
```
    python script/generate.py model=llava_7b_DRO dataset=FigStep +dataset.sample=4 debug=True
```

#### 8. Visualize the Hidden States at different token positions
```
    python script/HiddenStates/analyze_hidden_states.py --config-name hidden_states.yaml 
```





#### 8. Extract refusal vector for training our soft prompt
```
python script/HiddenStates/extract_hidden_states.py --config-name hidden_states_train.yaml 
```



#### 9. GCG Attack
Training:
```
    python script/GCG/gcg.py --config-name GCG.yaml  debug=True
    # python script/GCG/gcg.py --config-name GCG.yaml  model=Qwen2VL debug=True
```
Testing:
```
    python script/generate_text_only.py model=llava_7b dataset=GCG
```

#### 10. BAP Attack
Text-agnostic adv image generation
```
    python script/BAP/adv_img_train_llava.py
    # python script/BAP/adv_img_train_Qwen2VL.py
```

Bi-model adv samples generation
```
    python script/BAP/BiModa_attack.py debug=True
    #     python script/BAP/BiModa_attack.py model=Qwen2VL debug=True 
```
Attack the VLMs
```
     python script/BAP/attack.py model=llava_7b
     #  python script/BAP/attack.py model=Qwen2VL
```
