import json

input_path = "response.json"
output_path = "MMVet.json"

with open(input_path, "r") as f:
    raw_data = json.load(f)

data = {}

for i in range(len(raw_data)):
    index = 'v1_' + str(i)
    data[index] = raw_data[i]['response']

with open(output_path, "w") as f:
    json.dump(data, f, indent=4)