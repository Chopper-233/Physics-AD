import json
import os

result = []
out_dir = "scores"
for json_result in os.listdir(out_dir):
    with open(os.path.join(out_dir, json_result)) as j:
        result.append((json_result, json.load(j)))

task = ['describ','explain']

with open("result.txt",'w') as f:
    for item in result:
        name = item[0][:-5]
        re = item[1]
        for t in task:
            score = 0
            for obj in re.keys():
                if t in re[obj].keys(): score+=re[obj][t]
            score /= len(re.keys())
            print(f"{name}, {t}, {score}",file=f)

