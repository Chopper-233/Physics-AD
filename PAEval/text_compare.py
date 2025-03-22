import argparse
import json
import os
import sys
from pathlib import Path
import base64
from openai import OpenAI
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

key = ''

class Gpt4oModel(nn.Module):
    def __init__(self, args):
        self.client = OpenAI(api_key=key, base_url="https://api.chatanywhere.tech/v1")
    
    def chat(self, text:str, label:list):        
        conversation = self.client.chat.completions.create(
            model = "gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content":"I am an expert in text comparison. I evaluate the similarity between texts by leveraging semantic information, including spatiotemporal relationships between concepts and various elements of events. I then derive a score between 0 and 1, with higher scores indicating greater similarity."
                },
                {
                    "role":"user",
                    "content":f"Here is an input text that needs to be scored: {text}. Compare this input text with each entry in the label text library: {label}. Assign a similarity score, and output the highest score as the final result. Just output the final result and no other reply."    
                }
            ]
        )
        return conversation.choices[0].message.content

    
def run_with_json(
    des_dict,
    exp_dict,
    label_dict,
    output_file,
    task_select='both',
    spec_obj=None,
    
): 
    compare_spec = Gpt4oModel(args)
    scores = {}
    tasks = ['descrip','explain']
    choices = ['description_label', 'explanation_label']
    answerlist = [des_dict, exp_dict]
    for obj in label_dict.keys():
        if spec_obj == None or obj == spec_obj:
            print(obj)
            scores[obj]={}

            if task_select == 'both': task_id = [0,1] 
            elif task_select == 'describ': task_id = [0]
            elif task_select == 'explain': task_id = [1]
            else: 
                print("error task")
                break

            for i in task_id:
                scores[obj][tasks[i]]={}
                for abn in answerlist[i][obj].keys():
                    scores[obj][tasks[i]][abn]=[]
                    text_list = answerlist[i][obj][abn]
                    label_list = label_dict[obj][choices[i]][abn]
                    for text_query in tqdm(text_list):
                        answer = compare_spec.chat(text=text_query, label=label_list)
                        scores[obj][tasks[i]][abn].append(answer)

    with open(f"{output_file}",'w') as j:
        json.dump(scores,j,indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dict", type=str, default=r"label_lib.json")
    parser.add_argument("--output_file", type=str, default=r"raw_output\sample_output.json")
    parser.add_argument("--task", type=str, default="both", help="describ or explain or both")
    parser.add_argument("--despath", type=str, default=r"results_from_VLM\des_sample.json", help="descrip file path")
    parser.add_argument("--exppath", type=str, default=r"results_from_VLM\exp_sample.json", help="explain file path")
    parser.add_argument("--obj", type=str, default=None, help="if you only want to test one object")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.label_dict,'r', encoding='utf-8') as j:
        label_dict = json.load(j)

    if args.despath:
        with open(args.despath,'r', encoding='utf-8') as d:
            descrip_dict = json.load(d)
    else: descrip_dict = None

    if args.exppath:
        with open(args.exppath,'r', encoding='utf-8') as e:
            explain_dict = json.load(e)
    else: explain_dict = None
    
    run_with_json(
        des_dict=descrip_dict,
        exp_dict=explain_dict,
        label_dict=label_dict,
        output_file=args.output_file,
        task_select=args.task,
        spec_obj=args.obj
    )
