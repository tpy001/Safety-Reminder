# export HF_ENDPOINT=https://hf-mirror.com

git clone https://github.com/deepseek-ai/DeepSeek-VL.git
cd DeepSeek-VL
pip install -e .

cd ..
huggingface-cli download deepseek-ai/deepseek-vl-7b-chat --local-dir hf_models/DeepSeek_VL

