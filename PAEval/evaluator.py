import json
import re
import os

scoresofall = []
out_dir = "raw_output"
for json_result in os.listdir(out_dir):
    with open(os.path.join(out_dir, json_result)) as j:
        scoresofall.append((json_result, json.load(j)))

for item in scoresofall:
    name = item[0][:-5]
    scores = item[1]
    e = {}
    for obj in scores.keys():
        e[obj]={}
        for task in scores[obj].keys():
            
            avg_score = 0
            avg_num = 0
            for abn in scores[obj][task].keys():
                for entry in scores[obj][task][abn]:
                    numbers = re.findall(r'0?\.\d+|1\.0*|0', entry)
                    fin_score = float(numbers[-1])
                    # fin_score = float(entry)
                    avg_score += fin_score
                    avg_num += 1
            avg_score /= avg_num
            e[obj][task] = avg_score

    with open(f"scores/eval_{name}.json",'w') as j:
        json.dump(e, j, indent=4)