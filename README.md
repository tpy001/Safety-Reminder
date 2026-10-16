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

LLaVA-1.5-7B results reported in the [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/40607) (Tables 1 and 2). Bold values indicate the best result in each column.

### Safety: attack success rate (%)

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">Harmful benchmarks &darr;</th>
      <th colspan="3" align="center">Adversarial attacks &darr;</th>
      <th rowspan="2" align="center">Avg. &darr;</th>
    </tr>
    <tr>
      <th align="center">FigStep</th>
      <th align="center">MMSafety</th>
      <th align="center">SPA-VL</th>
      <th align="center">ImgJP</th>
      <th align="center">GCG</th>
      <th align="center">BAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">No Defense</td>
      <td align="center">74.9</td>
      <td align="center">58.2</td>
      <td align="center">45.6</td>
      <td align="center">86.6</td>
      <td align="center">94.0</td>
      <td align="center">97.0</td>
      <td align="center">76.1</td>
    </tr>
    <tr>
      <td align="left">Self-Reminder</td>
      <td align="center">72.0</td>
      <td align="center">50.0</td>
      <td align="center">9.8</td>
      <td align="center">85.2</td>
      <td align="center">51.0</td>
      <td align="center">97.0</td>
      <td align="center">60.8</td>
    </tr>
    <tr>
      <td align="left">DRO</td>
      <td align="center">69.7</td>
      <td align="center">47.0</td>
      <td align="center">18.1</td>
      <td align="center">83.2</td>
      <td align="center">64.0</td>
      <td align="center">90.0</td>
      <td align="center">62.0</td>
    </tr>
    <tr>
      <td align="left">BlueSuffix</td>
      <td align="center">72.3</td>
      <td align="center">39.8</td>
      <td align="center">8.3</td>
      <td align="center">38.4</td>
      <td align="center">53.0</td>
      <td align="center">43.0</td>
      <td align="center">42.5</td>
    </tr>
    <tr>
      <td align="left">ASTRA</td>
      <td align="center">8.6</td>
      <td align="center">25.9</td>
      <td align="center">30.9</td>
      <td align="center"><strong>3.4</strong></td>
      <td align="center">6.0</td>
      <td align="center">5.0</td>
      <td align="center">13.3</td>
    </tr>
    <tr>
      <td align="left"><strong>SAPT (Ours)</strong></td>
      <td align="center"><strong>2.6</strong></td>
      <td align="center"><strong>12.7</strong></td>
      <td align="center"><strong>1.5</strong></td>
      <td align="center">4.4</td>
      <td align="center"><strong>5.0</strong></td>
      <td align="center"><strong>4.0</strong></td>
      <td align="center"><strong>5.0</strong></td>
    </tr>
  </tbody>
</table>

### Utility: MM-Vet

<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">Overall score &uarr;</th>
      <th align="center">Refusal rate (%) &darr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">No Defense</td>
      <td align="center">27.9</td>
      <td align="center">2.3</td>
    </tr>
    <tr>
      <td align="left">Self-Reminder</td>
      <td align="center">27.6</td>
      <td align="center"><strong>1.4</strong></td>
    </tr>
    <tr>
      <td align="left">DRO</td>
      <td align="center"><strong>29.0</strong></td>
      <td align="center">2.8</td>
    </tr>
    <tr>
      <td align="left">BlueSuffix</td>
      <td align="center">15.6</td>
      <td align="center">7.8</td>
    </tr>
    <tr>
      <td align="left">ASTRA</td>
      <td align="center">27.8</td>
      <td align="center">1.8</td>
    </tr>
    <tr>
      <td align="left"><strong>SAPT (Ours)</strong></td>
      <td align="center">28.0</td>
      <td align="center">3.7</td>
    </tr>
  </tbody>
</table>

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

## Citation

If you find this repository helpful, please cite our paper:

```bibtex
@article{tang2026safetyreminder,
  title   = {SafetyReminder: Reviving Delayed Safety Awareness of Vision-Language Models to Defend Against Jailbreak Attacks},
  author  = {Tang, Peiyuan and Xin, Haojie and Zhang, Xiaodong and Sun, Jun and Xia, Qin and Yang, Zijiang James},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {39},
  pages   = {33223--33231},
  year    = {2026},
  doi     = {10.1609/aaai.v40i39.40607},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/40607}
}
```
