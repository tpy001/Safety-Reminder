# SafetyReminder

Implementation of [**SafetyReminder: Reviving Delayed Safety Awareness of Vision-Language Models to Defend Against Jailbreak Attacks**](https://ojs.aaai.org/index.php/AAAI/article/view/40607/44568), AAAI 2026.

This branch provides LLaVA-1.5-7B baseline inference, Safety-Activation Prompt Tuning (SAPT), and inference with the trained safety reminder.

## Core idea

A jailbroken VLM may initially produce harmful content but recognize the risk later in its response. SafetyReminder aims to activate this **delayed safety awareness earlier**, allowing the model to switch to a safe response during generation.

**Training:** SAPT learns four soft prompt embeddings from a balanced mixture of harmful and benign examples. For harmful examples, the prompt follows an unsafe response prefix and guides the model toward a safe answer; for benign examples, it preserves helpful continuation. The objective combines **text generation loss** with **hidden state supervision** through a frozen safety-awareness classifier: `L = L_gen + 0.05 * L_cls`. Only the soft prompt is updated; the base model and classifier remain frozen.

**Inference:** Every 16 decoding steps, temporarily append the learned prompt and inspect its final hidden state with the classifier. If a risk is detected, retain the prompt and start a safe continuation; otherwise, remove it and continue normal decoding. This conditional injection applies the reminder when needed while limiting interference with benign responses.

![SAPT framework from the AAAI 2026 paper](assets/SAPT.png)

*Figure 3 from the AAAI 2026 paper: SAPT training and conditional soft prompt injection.*

## Results

LLaVA-1.5-7B results reported in the AAAI 2026 paper (Tables 1 and 2), not new measurements from this branch.

**Attack success rate (ASR, %, lower is better)**

| Method | FigStep | MMSafety | SPA-VL | ImgJP | GCG | BAP | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No Defense | 74.9 | 58.2 | 45.6 | 86.6 | 94.0 | 97.0 | 76.1 |
| Self-Reminder | 72.0 | 50.0 | 9.8 | 85.2 | 51.0 | 97.0 | 60.8 |
| DRO | 69.7 | 47.0 | 18.1 | 83.2 | 64.0 | 90.0 | 62.0 |
| BlueSuffix | 72.3 | 39.8 | 8.3 | 38.4 | 53.0 | 43.0 | 42.5 |
| ASTRA | 8.6 | 25.9 | 30.9 | **3.4** | 6.0 | 5.0 | 13.3 |
| SAPT | **2.6** | **12.7** | **1.5** | 4.4 | **5.0** | **4.0** | **5.0** |

**MM-Vet utility**

| Method | Overall score (higher is better) | Refusal rate (%, lower is better) |
| --- | ---: | ---: |
| No Defense | 27.9 | 2.3 |
| Self-Reminder | 27.6 | **1.4** |
| DRO | **29.0** | 2.8 |
| BlueSuffix | 15.6 | 7.8 |
| ASTRA | 27.8 | 1.8 |
| SAPT | 28.0 | 3.7 |

## 1. Installation

```bash
conda create -n safevlm python=3.10 -y
conda activate safevlm
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

## 2. Download models and datasets

```bash
hf download llava-hf/llava-1.5-7b-hf --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 --local-dir models/llava-1.5-7b
hf download PeiYuan001/VLM-Safety-Datasets --repo-type dataset --revision 48cf6aa39472c962ddd0b2d627cbf23ea5023b48 --include "data/*" --local-dir datasets/vlm-safety
# Requires access to the Llama-Guard-3 model
hf download meta-llama/Llama-Guard-3-8B --local-dir models/Llama-Guard-3-8B
```

## 3. Dataset inference and safety evaluation

```bash
DATASET=spavl_harm  # spavl_harm, figstep, or mmsafety_sd_typo
for MODE in baseline sapt; do
  python -m safevlm --mode "$MODE" --dataset-config "$DATASET" --data-dir datasets/vlm-safety --device cuda:0 --output "outputs/${DATASET}_${MODE}.jsonl"
  python -m safevlm.judge --model-path models/Llama-Guard-3-8B --device cuda:0 --input "outputs/${DATASET}_${MODE}.jsonl" --output "outputs/${DATASET}_${MODE}_judged.jsonl"
done
cat outputs/${DATASET}_*_judged.summary.json
```

Each summary reports `asr_percent = 100 * unsafe / n`, plus `safe` and `unknown` counts. Unknown labels remain in the denominator. The commands evaluate the entire selected configuration.

## 4. Train SAPT

```bash
# 256 harmful + 256 benign; L = L_gen + 0.05 * L_cls
python -m safevlm.train --data-dir datasets/vlm-safety --safety-pool legacy --samples-per-pool 256 --epochs 20 --lr 0.0001 --grad-accum 4 --seed 0 --device cuda:0 --output-dir outputs/train_sapt
```

To evaluate the trained prompt, add `--prompt-path outputs/train_sapt/prompt.pth` to the SAPT inference command above.

## 5. MM-Vet utility evaluation

```bash
git clone https://github.com/yuweihao/MM-Vet_data.git datasets/mmvet
git -C datasets/mmvet checkout 2df56155a024929e29ebc01efe288d67eca17f62
python -m safevlm.mmvet prepare --source datasets/mmvet --output outputs/mmvet/input.jsonl
python -m safevlm --mode baseline --input outputs/mmvet/input.jsonl --output outputs/mmvet/baseline.jsonl --mmvet-source datasets/mmvet
python -m safevlm --mode sapt --input outputs/mmvet/input.jsonl --output outputs/mmvet/sapt.jsonl --mmvet-source datasets/mmvet
```

Submit `outputs/mmvet/baseline.mmvet.json` and `outputs/mmvet/sapt.mmvet.json` to the [MM-Vet v1 evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).
