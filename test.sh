  # python script/generate.py dataset=MMSafetyBench,SPA_VL,MMVet model=DeepSeekVL --multirun 
  # python script/generate.py --multirun  dataset=PGDAttackDataset model=DeepSeekVL debug=True
# python script/generate.py --multirun dataset=FigStep,MMSafetyBench,PGDAttackDataset,MMVet model=DeepSeekVL_DRO # Sample 4 data from the dataset
# python script/generate.py dataset=FigStep model=DeepSeekVL_SAPT debug=True

#  python script/generate.py dataset=MMSafetyBench_benign
# python script/HiddenStates/analyze_hidden_states_new_Qwen2.py --config-name hidden_states model=Qwen2VL

# python script/generate_test.py dataset=PGDAttackDataset model=Qwen2VL debug=True # Sample 4 data from the dataset

# python script/HiddenStates/analyze_hidden_states_new.py --config-name hidden_states model=llava_7b_SAPT debug=True

# python script/generate.py --multirun model=llava_7b_SAPT dataset=FigStep,MMSafetyBench,PGDAttackDataset,MMVet
 
# python script/generate.py --multirun model=llava_7b_SAPT dataset=PGDAttackDataset debug=True

# python script/generate.py --multirun model=llava_13b_SAPT dataset=FigStep debug=True

# python script/generate.py model=llava_7b_SAPT dataset=MMVet debug=True

# python script/generate.py model=DeepSeekVL_SAPT dataset=PGDAttackDataset debug=True

python script/generate.py model=llava_7b dataset=SPA_VL debug=True


