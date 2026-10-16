# SafetyReminder

Implementation of [**SafetyReminder: Reviving Delayed Safety Awareness of Vision-Language Models to Defend Against Jailbreak Attacks**](https://ojs.aaai.org/index.php/AAAI/article/view/40607/44568), AAAI 2026.

This branch provides LLaVA-1.5-7B baseline inference, Safety-Activation Prompt Tuning (SAPT), and inference with the trained safety reminder.

## Core idea

A jailbroken VLM may initially produce harmful content but recognize the risk later in its response. SafetyReminder aims to activate this **delayed safety awareness earlier**, allowing the model to switch to a safe response during generation.

**Training:** SAPT learns four soft prompt embeddings from a balanced mixture of harmful and benign examples. For harmful examples, the prompt follows an unsafe response prefix and guides the model toward a safe answer; for benign examples, it preserves helpful continuation. The objective combines **text generation loss** with **hidden state supervision** through a frozen safety-awareness classifier: `L = L_gen + 0.05 * L_cls`. Only the soft prompt is updated; the base model and classifier remain frozen.

**Inference:** Every 16 decoding steps, temporarily append the learned prompt and inspect its final hidden state with the classifier. If a risk is detected, retain the prompt and start a safe continuation; otherwise, remove it and continue normal decoding. This conditional injection applies the reminder when needed while limiting interference with benign responses.

![SAPT framework from the AAAI 2026 paper](assets/SAPT.png)

*Figure 3 from the AAAI 2026 paper: SAPT training and conditional soft prompt injection.*

## 1. Installation

```bash
conda create -n safevlm python=3.10 -y
conda activate safevlm
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

## 2. Download model and datasets

```bash
hf download llava-hf/llava-1.5-7b-hf --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 --local-dir models/llava-1.5-7b
hf download PeiYuan001/VLM-Safety-Datasets --repo-type dataset --revision 48cf6aa39472c962ddd0b2d627cbf23ea5023b48 --include "data/*" --local-dir datasets/vlm-safety
```

## 3. Inference

```bash
# Text
python -m safevlm --mode baseline --device cuda:0 --question "Explain what a rainbow is."

# Image and question; SAPT weights and classifier are included
python -m safevlm --mode sapt --device cuda:0 --question "Describe this image." --image examples/sample.jpg

# JSONL; image paths are relative to the input file
python -m safevlm --mode sapt --input examples/requests.jsonl --output outputs/responses.jsonl

# The same SPA-VL subset for baseline and SAPT
python -m safevlm --mode baseline --dataset-config spavl_harm --data-dir datasets/vlm-safety --ids-file safevlm/resources/spavl_subset_ids.json --output outputs/spavl_baseline.jsonl
python -m safevlm --mode sapt --dataset-config spavl_harm --data-dir datasets/vlm-safety --ids-file safevlm/resources/spavl_subset_ids.json --output outputs/spavl_sapt.jsonl

# Other benchmarks: figstep, mmsafety_sd, mmsafety_typo, mmsafety_sd_typo
python -m safevlm --mode sapt --dataset-config figstep --data-dir datasets/vlm-safety --limit 10 --output outputs/figstep_sapt.jsonl
```

## 4. Train SAPT

```bash
# One-update check with two harmful and two benign examples
python -m safevlm.train --data-dir datasets/vlm-safety --samples-per-pool 2 --epochs 1 --max-steps 1 --output-dir outputs/train_smoke

# 256 harmful + 256 benign; L = L_gen + 0.05 * L_cls
python -m safevlm.train --data-dir datasets/vlm-safety --safety-pool legacy --samples-per-pool 256 --epochs 20 --lr 0.0001 --grad-accum 4 --seed 0 --device cuda:0 --output-dir outputs/train_sapt

# Inference with the trained prompt
python -m safevlm --mode sapt --prompt-path outputs/train_sapt/prompt.pth --input examples/requests.jsonl --output outputs/trained_responses.jsonl
```

## 5. Safety evaluation

```bash
# Requires access to the Llama-Guard-3 model
hf download meta-llama/Llama-Guard-3-8B --local-dir models/Llama-Guard-3-8B
python -m safevlm.judge --model-path models/Llama-Guard-3-8B --input outputs/spavl_baseline.jsonl --output outputs/spavl_baseline_judged.jsonl
python -m safevlm.judge --model-path models/Llama-Guard-3-8B --input outputs/spavl_sapt.jsonl --output outputs/spavl_sapt_judged.jsonl
```

## 6. MM-Vet utility evaluation

```bash
git clone https://github.com/yuweihao/MM-Vet_data.git datasets/mmvet
git -C datasets/mmvet checkout 2df56155a024929e29ebc01efe288d67eca17f62
python -m safevlm.mmvet prepare --source datasets/mmvet --output outputs/mmvet/input.jsonl
python -m safevlm --mode baseline --input outputs/mmvet/input.jsonl --output outputs/mmvet/baseline.jsonl --mmvet-source datasets/mmvet
python -m safevlm --mode sapt --input outputs/mmvet/input.jsonl --output outputs/mmvet/sapt.jsonl --mmvet-source datasets/mmvet
```

Submit `outputs/mmvet/baseline.mmvet.json` and `outputs/mmvet/sapt.mmvet.json` to the [MM-Vet v1 evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).
