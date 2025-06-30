"""
Sample and preprocess data from the specified categories of interest.

Step 1: Define the categories of interest and the desired number of samples per category.  
Step 2: Filter the dataset to include only data belonging to the specified categories.  
Step 3: Use the Llama-Guard Vision model to filter out safe data (i.e., only keep unsafe data)..  
Step 4: Sample the filtered dataset according to the specified number of samples per category.  
Step 5: Save the preprocessed data to the designated data folder.
"""

from datasets import load_dataset
import json

categories = {  
    "Illegal Activity": [
        "Malicious Use/Illegal activities/drug-related crimes",
        "Malicious Use/Illegal activities/environmental crimes",
        "Malicious Use/Illegal activities/financial crimes",
        "Malicious Use/Illegal activities/theft",
        "Malicious Use/Illegal activities/traffic and driving offenses",
        "Malicious Use/Security threats/biological and chemical harms",
        "Malicious Use/Security threats/weapon",
    ],
    "Hate Speech": [
        "Representation & ToxicityHarms/Toxic/bully",
        "Representation & ToxicityHarms/Toxic/harass",
        "Representation & ToxicityHarms/Toxic/hate",
        "Representation & ToxicityHarms/Toxic/insult",
        "Representation & ToxicityHarms/Toxic/terrorism",
        "Representation & ToxicityHarms/Toxic/threaten",
        "Representation & ToxicityHarms/Toxic/violent",
        "Representation & ToxicityHarms/Unfair/ageism",
        "Representation & ToxicityHarms/Unfair/disablity discrimination",
        "Representation & ToxicityHarms/Unfair/geographical discrimination",
        "Representation & ToxicityHarms/Unfair/racial",
        "Representation & ToxicityHarms/Unfair/religion",
        "Representation & ToxicityHarms/Unfair/sexism",
    ],
    "Malware Generation":[
        "Malicious Use/Security threats/malware",
        "Malicious Use/Security threats/cyber",
    ],
    "physical harm": [
        "Human Autonomy & IntegrityHarms/Persuasion and manipulation/physical harm",
    ],
    "Economic Harm": [
        "Socioeconomic Harms/Risky financial practices/gamble",
        "Socioeconomic Harms/Risky financial practices/multi-level marketing",
        "Socioeconomic Harms/Risky financial practices/paypal lending",
    ],
    "Fraud": [
        "Malicious Use/Fraud or deceptive action/academic dishonesty",
        "Malicious Use/Fraud or deceptive action/disinformation",
        "Malicious Use/Fraud or deceptive action/fake news",
        "Malicious Use/Fraud or deceptive action/financial fraud",
        "Malicious Use/Fraud or deceptive action/pseudo-pharmaceuticals",
        "Malicious Use/Fraud or deceptive action/rumors",
        "Malicious Use/Fraud or deceptive action/scamns",
        "Malicious Use/Fraud or deceptive action/spam information",
    ],
    "Privacy Violation": [
        "Information & Safety Harms/Privacy/personal privacy"
    ],
    "Adult Content": [

    ]
}

# sample total 1000 data
categories_nums = {
    "Illegal Activity": 200,
    "Hate Speech": 150,
    "Malware Generation": 150,
    "physical harm": 100,
    "Economic Harm": 100,
    "Fraud": 200,
    "Privacy Violation": 100,
    "Adult Content": 0
}


with open('data/SPA_VL/train/meta.json', 'r') as f:
    dataset = json.load(f)

print(dataset[0])



