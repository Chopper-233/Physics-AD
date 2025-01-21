"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
from options.VideoLLaMA.option import Options
import os
import random
import csv
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%

# 通用问题（所有类别共享的四个问题）
common_questions = [
    "What is the object in the video?",
    "What is the normal function of the object in the video in real life?",
    "What is the mode of interaction in the video?",
    "Please describe the content of this video, focusing on aspects such as objects, appearance, psysics interaction and so on.",
    "Assume you are an anomaly detection expert,do you think the fuction of the object in the video is normal or abnormal,please give me a reasonable explanation.",
]

# 针对每个类别的特定问题
category_specific_questions = {
    'ball': "Assume you are an anomaly detection expert, the object in video is a ball,under normal circumstances, when a ball is fully inflated, it is difficult to deform significantly. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'botton': "Assume you are an anomaly detection expert, the object in video is a button that can be pressed normally and springs back, with a light that turns on when pressed. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'car': "Assume you are an anomaly detection expert, the object in video is a toy car,under normal circumstances, the front and rear wheels of a toy car can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'caster_wheel': "Assume you are an anomaly detection expert, the object in video is a caster_wheel,Under normal circumstances, the rolling axis and the universal axis of a caster wheel can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'clip': "Assume you are an anomaly detection expert, the object in video is a clip,Under normal circumstances, a clip can be pressed down to a certain angle and then rebound back to its original position. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'clock': "Assume you are an anomaly detection expert, the object in video is a clock,Under normal circumstances, the second hand of a clock can rotate.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'fan': "Assume you are an anomaly detection expert, the object in video is a fan,Under normal circumstances, the fan blades can rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'gear': "Assume you are an anomaly detection expert, the object in video is a gear,Under normal circumstances, all gears rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'hinge': "Assume you are an anomaly detection expert, the object in video is a hinge,Under normal circumstances, the hinge can rotate without angle limitations and does not come loose.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'liguid': "Assume you are an anomaly detection expert, the object in video is a bottle with liquid,Under normal circumstances, the liquid is free of foreign objects and does not leak.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'lock': "Assume you are an anomaly detection expert, the object in video is a door lock,Under normal circumstances, when the lock is turned, the bolt can retract properly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'magnet': "Assume you are an anomaly detection expert, the object in video is a magnet,Under normal circumstances, a magnet can stick to iron.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'rolling_bear': "Assume you are an anomaly detection expert, the object in video is rolling_bear,Under normal circumstances, the balls inside a bearing can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'rubber_band': "Assume you are an anomaly detection expert, the object in video is rubber band,Under normal circumstances, a rubber band can stretch without cracking or breaking.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'screw': "Assume you are an anomaly detection expert, the object in video is screw,Under normal circumstances, a screw can be tightened securely.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'servo': "Assume you are an anomaly detection expert, the object in video is servo,Under normal circumstances, a servo motor can rotate without angle limitations and return to its center position.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'slide': "Assume you are an anomaly detection expert, the object in video is slide,Under normal circumstances, the sliding rail can move smoothly, and the structure remains intact.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'spherical_bearing': "Assume you are an anomaly detection expert, the object in video is spherical_bearing,Under normal circumstances, a spherical bearing can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'sticky_roller': "Assume you are an anomaly detection expert, the object in video is sticky_roller,Under normal circumstances, the roller can rotate smoothly without falling off.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'toothpaste': "Assume you are an anomaly detection expert, the object in video is toothpaste,Under normal circumstances, toothpaste does not leak when squeezed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'usb': "Assume you are an anomaly detection expert, the object in video is usb,Under normal circumstances, the USB cap can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",
    'zipper': "Assume you are an anomaly detection expert, the object in video is zipper,Under normal circumstances, a zipper can open and close smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= }.",

}

# 保存 CSV 的路径
output_path = 'results/VideoLLaMA'  # 需要替换为你的保存路径

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
parser = Options()
args = parser.parse_args()

cfg = Config(args)
print(cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id

print('Model architecture specified in the configuration:', model_config.arch)
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# 这是你的视频目录路径
base_path =  args.dataset_path
# 假设你要问的五个问题
general_questions = [
    "What is the object in the video?",
   "What is the normal function of the object in the video in real life?",
    "What is the mode of interaction in the video?",
    "Please describe the content of this video, focusing on aspects such as objects, appearance ,psysics interaction and so on.",
    "Assume you are an anomaly detection expert,do you think the fuction of the object in the video is normal or abnormal,please give me a reasonable explanation.",
]

# 针对每个类别的特定问题
category_specific_questions = {
    'ball': "Assume you are an anomaly detection expert, the object in video is a ball,under normal circumstances, when a ball is fully inflated, it is difficult to deform significantly. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'button': "Assume you are an anomaly detection expert, the object in video is a button that can be pressed normally and springs back, with a light that turns on when pressed. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'car': "Assume you are an anomaly detection expert, the object in video is a toy car,under normal circumstances, the front and rear wheels of a toy car can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'caster_wheel': "Assume you are an anomaly detection expert, the object in video is a caster_wheel,Under normal circumstances, the rolling axis and the universal axis of a caster wheel can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words..",
    'clip': "Assume you are an anomaly detection expert, the object in video is a clip,Under normal circumstances, a clip can be pressed down to a certain angle and then rebound back to its original position. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'clock': "Assume you are an anomaly detection expert, the object in video is a clock,Under normal circumstances, the second hand of a clock can rotate.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'fan': "Assume you are an anomaly detection expert, the object in video is a fan,Under normal circumstances, the fan blades can rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'gear': "Assume you are an anomaly detection expert, the object in video is a gear,Under normal circumstances, all gears rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'hinge': "Assume you are an anomaly detection expert, the object in video is a hinge,Under normal circumstances, the hinge can rotate without angle limitations and does not come loose.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'liquid': "Assume you are an anomaly detection expert, the object in video is a bottle with liquid,Under normal circumstances, the liquid is free of foreign objects and does not leak.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'lock': "Assume you are an anomaly detection expert, the object in video is a door lock,Under normal circumstances, when the lock is turned, the bolt can retract properly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'magnet': "Assume you are an anomaly detection expert, the object in video is a magnet,Under normal circumstances, a magnet can stick to iron.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'rolling_bearing': "Assume you are an anomaly detection expert, the object in video is rolling_bearing,Under normal circumstances, the balls inside a bearing can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'rubber_band': "Assume you are an anomaly detection expert, the object in video is rubber band,Under normal circumstances, a rubber band can stretch without cracking or breaking.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'screw': "Assume you are an anomaly detection expert, the object in video is screw,Under normal circumstances, a screw can be tightened securely.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'servo': "Assume you are an anomaly detection expert, the object in video is servo,Under normal circumstances, a servo motor can rotate without angle limitations and return to its center position.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'slide': "Assume you are an anomaly detection expert, the object in video is slide,Under normal circumstances, the sliding rail can move smoothly, and the structure remains intact.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'spherical_bearing': "Assume you are an anomaly detection expert, the object in video is spherical_bearing,Under normal circumstances, a spherical bearing can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'sticky_roller': "Assume you are an anomaly detection expert, the object in video is sticky_roller,Under normal circumstances, the roller can rotate smoothly without falling off.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'toothpaste': "Assume you are an anomaly detection expert, the object in video is toothpaste,Under normal circumstances, toothpaste does not leak when squeezed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'usb': "Assume you are an anomaly detection expert, the object in video is usb,Under normal circumstances, the USB cap can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",
    'zipper': "Assume you are an anomaly detection expert, the object in video is zipper,Under normal circumstances, a zipper can open and close smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: '{anomalyscore= }'without any other words.",

}

def ask_questions(video_path, general_questions, specific_question, chat, chat_state):
    # 这个函数假设你已经有chat和chat_state
    img_list = []  # 假设处理视频的img_list需要初始化为空
    responses = []

    # 上传视频
    chat.upload_video(video_path, chat_state, img_list)
    
    # 针对每个通用问题进行提问
    for question in general_questions:
        chat.ask(question, chat_state)
        response = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=20000
        )[0]
        responses.append(response)
    
    # 针对特定问题进行提问
    chat.ask(specific_question, chat_state)
    specific_response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=20000
    )[0]
    responses.append(specific_response)
    
    return responses

# 保存到CSV的函数
def save_to_csv(category, video_path, answers, defect_type):
    csv_save_path=os.path.join(output_path,"csvs")
    
    csv_path = os.path.join(csv_save_path,f"{category}", f"{defect_type}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 检查CSV文件是否存在，不存在则创建并写入表头
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # 写入表头
            writer.writerow(['Video Path', 'Answer 1', 'Answer 2', 'Answer 3', 'Answer 4', 'Specific Answer'])
        # 写入视频路径及其回答
        writer.writerow([video_path] + answers)
        
    specific_csv_path = os.path.join(csv_save_path, f"{category}",f"{defect_type}_score.csv")
    specific_exists = os.path.isfile(specific_csv_path)
    with open(specific_csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not specific_exists:
            # 写入表头
            writer.writerow(['Video Path', 'AS'])
        # 仅写入视频路径和特定问题的回答
        writer.writerow([video_path, answers[-1]])
        
def save_to_txt(category, video_path, answers, defect_type):
    # 设置保存 .txt 文件的基础路径
    txt_save_path = os.path.join(output_path, "txts")
    
    # 文件 1：包含路径和所有回答
    all_answers_path = os.path.join(txt_save_path, f"{category}", f"{defect_type}_all_answers.txt")
    os.makedirs(os.path.dirname(all_answers_path), exist_ok=True)

    # 文件 2：仅包含路径和最后一个回答
    specific_answer_path = os.path.join(txt_save_path, f"{category}", f"{defect_type}_explain_answer.txt")
    os.makedirs(os.path.dirname(specific_answer_path), exist_ok=True)
    
    describe_answer_path = os.path.join(txt_save_path, f"{category}", f"{defect_type}_descrip_answer.txt")
    os.makedirs(os.path.dirname(describe_answer_path), exist_ok=True)

    # 写入文件 1：路径和所有回答
    with open(all_answers_path, mode='a') as file_full, open(specific_answer_path, mode='a') as file_explain, open(describe_answer_path, mode='a') as file_describe:

        for i, answer in enumerate(answers, 1):  # 写入每个回答
            file_full.write(f"Video Path: {video_path}\nanswer {i}: {answer}\n")
            if i==4: 
                file_describe.write(f"Video Path: {video_path}\nanswer {i}: {answer}\n\n")
            if i==5:
                file_explain.write(f"Video Path: {video_path}\nanswer {i}: {answer}\n\n")
        file_full.write("\n")  # 添加空行分隔不同记录
    print(f"All answers written to {all_answers_path}")

    

# 遍历目录并处理视频

category_path = os.path.join(base_path, args.obj)

# 确保是一个目录
if os.path.isdir(category_path):
    # 进入第二级目录，寻找'test'目录
    test_path = os.path.join(category_path, 'test')
    
    if os.path.isdir(test_path):
        # 检查是否存在该类别的特定问题
        specific_question = category_specific_questions.get(args.obj, "Default specific question?")
        
        # 遍历'test'目录下的缺陷类型子目录
        for defect_type in os.listdir(test_path):
            defect_path = os.path.join(test_path, defect_type)
            
            if os.path.isdir(defect_path):
                # 遍历缺陷类型子目录下的所有视频文件
                for video_file in os.listdir(defect_path):
                    video_path = os.path.join(defect_path, video_file)
                    
                    if os.path.isfile(video_path) and video_file.endswith(('.mp4')):
                        print(f"Processing: {video_path}")
                        
                        # 获取视频的回答
                        chat_state = default_conversation.copy()  # 确保每次处理视频时重置chat_state
                        answers = ask_questions(video_path, general_questions, specific_question, chat, chat_state)
                        print(answers)
                        
                        # 保存到CSV
                        #save_to_csv(args.obj, video_path, answers,defect_type)
                        save_to_txt(args.obj, video_path, answers,defect_type)


