import json
import os

descrip_path = "results/VideoLLaVA/result_describ.txt"
explain_path = "results/VideoLLaVA/result_reason.txt"

descrip = {}
explain = {}
with open(descrip_path, 'r')as f:
    for l in f:
        l=l.strip()
        obj = l.split("/")[0]
        abn = l.split("/")[2]
        if obj not in descrip:
            descrip[obj]={}
            if abn not in descrip[obj]:
                descrip[obj][abn]=[]
        else:
            if abn not in descrip[obj]:
                descrip[obj][abn]=[]
        
        descrip[obj][abn].append(l.split(":")[1])
    
with open(explain_path, 'r')as f:
    lines = f.readlines()
    start_index = [l for l in range(len(lines)) if ".mp4" in lines[l]]
    end_index = [l for l in range(len(lines)) if ".mp4" in lines[l]][1:]
    end_index.append(len(lines))
    text_list = [lines[start_index[i]:end_index[i]] for i in range(len(start_index))]
    text_list = [" ".join(i).strip() for i in text_list]
    for l in text_list:
        l=l.strip()
        obj = l.split("/")[0]
        abn = l.split("/")[2]
        if obj not in explain:
            explain[obj]={}
            if abn not in explain[obj]:
                explain[obj][abn]=[]
        else:
            if abn not in explain[obj]:
                explain[obj][abn]=[]
        
        explain[obj][abn].append(l.split(":")[1])

with open("descrip.json", 'w') as d:
    json.dump(descrip, d, indent=4)

with open("explain.json", 'w') as e:
    json.dump(explain, e, indent=4)