#!/bin/bash

# export HF_ENDPOINT=https://hf-mirror.com

#Because downloading Llama-series models requires Meta’s approval, you must obtain the approval first and then log in to your Hugging Face account.

huggingface-cli login --token <your huggingface access token begin with 'hf_xxxxx'> #  replace the token with your huggingface access token

huggingface-cli download meta-llama/Llama-Guard-3-8B --local-dir hf_models/Llama3-Guard-8b
