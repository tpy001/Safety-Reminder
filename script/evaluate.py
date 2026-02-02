from src.evaluator import MDJudge,LLamaGuard3 
from src.utils import save_jailbreak_response
import os


import debugpy
print("Waiting for debugger...")
debugpy.listen(5678)
debugpy.wait_for_client()


# model_path = "weights/MD-judge-7b/"
# evaluator  = MDJudge(model_path)

model_path = "weights/Llama3-Guard-8b/"
evaluator  = LLamaGuard3(model_path)


data_file = "genereated_response.json"
import json 
with open(data_file, "r") as f:
    data = json.load(f)

questions = [ data[i]["question"] for i in range(len(data))]
response =   [ data[i]["response"] for i in range(len(data))]

_data = {
    "question":questions
}

safety_pred = evaluator.judge(response,_data)

save_jailbreak_response(response, _data, safety_pred,output_file_path="./genereated_response_new.json")



