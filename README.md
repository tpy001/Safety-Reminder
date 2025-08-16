# SafetyReminder

This is the official repository of 

**The Safety Reminder: A Soft Prompt to Reactivate Delayed Safety Awareness in Vision-Language Models**

*Peiyuan Tang\*,Haojie Xin\*,Xiaodong Zhang,Jun Sun,Qin Xia and Zijiang Yang*, arXiv 2025

<a href='https://arxiv.org/pdf/2506.15734' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>

## Abstract
As Vision-Language Models (VLMs) demonstrate increasing capabilities across real-world applications such as code generation and chatbot assistance, ensuring their safety has become paramount. Unlike traditional Large Language Models (LLMs), VLMs face unique vulnerabilities due to their multimodal nature, allowing adversaries to modify visual or textual inputs to bypass safety guardrails and trigger the generation of harmful content. Through systematic analysis of VLM behavior under attack, we identify a novel phenomenon termed “delayed safety awareness”. Specifically, we observe that safety-aligned VLMs may initially be compromised to produce harmful content, but eventually recognize the associated risks and attempt to self-correct. This pattern suggests that VLMs retain their underlying safety awareness but experience a temporal delay in their activation. Building on this insight, we hypothesize that VLMs’ safety awareness can be proactively reactivated through carefully designed prompts. To this end, we introduce “The Safety Reminder”, a soft prompt tuning approach that optimizes learnable prompt
tokens, which are periodically injected during the text generation process to enhance safety awareness, effectively preventing harmful content generation. Additionally, our safety reminder only activates when harmful content is detected, leaving normal
conversations unaffected and preserving the model’s performance on benign tasks. Through comprehensive evaluation across three established safety benchmarks and one adversarial attacks, we demonstrate that our approach significantly reduces attack success rates while maintaining model utility, offering a practical solution for deploying safer VLMs in real-world applications.


<figure style="text-align: center;">
  <img src="assets/Delayed_awareness.png" alt="Delayed Safety Awareness in VLMs" width="600">
  <figcaption><b>Figure 1:</b> Delayed Safety Awareness in VLMs</figcaption>
</figure>


<figure style="text-align: center;">
  <img src="assets/SAPT.png" alt="SAPT method">
  <figcaption><b>Figure 2:</b> We aim to address this problem by training a continuous soft prompt, called a safety reminder, to proactively reactivate safety awareness during text generation.</figcaption>
</figure>

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

#### 3. SPA-VL
Please download the dataset from https://huggingface.co/datasets/sqrti/SPA-VL



## Scripts
## 1. Text Generation and Safety Evaluation
Basic Generation
```
# Run text generation with llava_7b model on FigStep dataset
python script/generate.py dataset=FigStep model=llava_7b

# Run text generation with our SAPT-enhanced llava_7b model on FigStep dataset
python script/generate.py dataset=FigStep model=llava_7b_SAPT

# Run text generation with our Qwen2VL model on MMSafetyBench dataset
python script/generate.py dataset=FigStep model=Qwen2VL

# Run text generation with our SAPT-enhanced Qwen2VL model on MMSafetyBench dataset
python script/generate.py dataset=FigStep model=Qwen2VL_SAPT
```

## 2. Training soft prompts with our SAPT
```
    # Train SAPT for llava_7b model
     python script/SAPT/train.py

    # Train SAPT for Qwen2VL-7B model
     python script/SAPT/train.py model=Qwen2VL_SAPT
```
Pretrained soft prompts can be found in `pretrained_models/`

## 3. Adversarial Attack Methods
#### （1） ImgJP Attack
```
# for LlaVA model
python script/Jailbreak/LlaVA_PGD_attack.py

# for Qwen2VL model
python script/Jailbreak/Qwen2VL_PGD_attack.py
```
Pretrained adversarial images can be found in `data/PGD_attack`

#### （2） GCG Attack
```
# GCG Attack for Llava-1.5-7B model
python script/GCG/gcg.py --config-name GCG.yaml model=llava_7b

#  GCG Attack for Qwen2VL model
python script/GCG/gcg.py --config-name GCG.yaml model=Qwen2VL
```

Pretrained adversarial suffix can be found in `data/GCG`


#### （3） BAP Attack
```
# Step1: Text-agnostic adv image generation
  python script/BAP/adv_img_train_llava.py
# Step2: Bi-modal adv image generation
 python script/BAP/BiModa_attack.py 
# Step3: Attack the VLMs
 python script/BAP/attack.py model=llava_7b
```

Pretrained adversarial image and queris can be found in `data/BAP`

<!-- ## 4. Results on Different Datasets
The experiment results of different models on different datasets can be found in `results/` -->

## 4. Some Important Files:

- src/dataset: Implementation of various datasets
- src/model: Implementation of various models
- src/model/SAPT.py, SAPT_Qwen.py: Main implementation for our SAPT method
- script/SAPT/train_classifier.py: Training a safety-awareness classifier
- src/runner/SAPT_runner.py: Training soft prompt using our SAPT




