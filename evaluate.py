from src.evaluator import MDJudge,LLamaGuard3, RuleBasedEvaluator
from src.utils import save_jailbreak_response
import os


# import debugpy
# print("Waiting for debugger...")
# debugpy.listen(5678)
# debugpy.wait_for_client()


# model_path = "weights/MD-judge-7b/"
# evaluator  = MDJudge(model_path)

# model_path = "weights/Llama3-Guard-8b/"
# evaluator  = LLamaGuard3(model_path)

evaluator = RuleBasedEvaluator()

# data_file = "genereated_response.json"
# import json 
# with open(data_file, "r") as f:
#     data = json.load(f)

# questions = [ data[i]["question"] for i in range(len(data))]
# response =   [ data[i]["response"] for i in range(len(data))]

# _data = {
#     "question":questions
# }

def debug():
    import debugpy
    print("Waiting for debugger...")
    debugpy.listen(5678)
    debugpy.wait_for_client()

debug() 

data_file = "MMVet.json"
import json 
with open(data_file, "r") as f:
    data = json.load(f)

question = [ item for key, item in data.items() ]
_data = {
    "question": ["Hello"] * len(question)
}

safety_pred = evaluator.judge(question,_data)

save_jailbreak_response(question, _data, safety_pred,output_file_path="./genereated_response_new.json")



