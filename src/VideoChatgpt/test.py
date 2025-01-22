import os
from options.VideoChatgpt.option import Options
import datetime
import json
import time
import gradio as gr
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.demo.gradio_patch import Chatbot as grChatbot
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.demo.template import tos_markdown, css, title, disclaimer
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *


import logging

logging.basicConfig(level=logging.WARNING)

logger = build_logger("gradio_web_server", "src/VideoChatgpt/log/gradio_web_server.log")

headers = {"User-Agent": "Video-ChatGPT"}

COMMON_QUESTIONS = [
    "What is the object in the video?",
    "What is the normal function of the object in the video in real life?",
    "What is the mode of interaction in the video?",
    "Please describe the content of this video, focusing on aspects such as objects, appearance, psysics interaction and so on.",
    "Assume you are an anomaly detection expert,do you think the fuction of the object in the video is normal or abnormal,please give me a reasonable explanation."
]
CATEGORY_SPECIFIC_QUESTIONS = {
    'ball': "Assume you are an anomaly detection expert, the object in video is a ball,under normal circumstances, when a ball is fully inflated, it is difficult to deform significantly. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'button': "Assume you are an anomaly detection expert, the object in video is a button that can be pressed normally and springs back, with a light that turns on when pressed. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'car': "Assume you are an anomaly detection expert, the object in video is a toy car,under normal circumstances, the front and rear wheels of a toy car can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'caster_wheel': "Assume you are an anomaly detection expert, the object in video is a caster_wheel,Under normal circumstances, the rolling axis and the universal axis of a caster wheel can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'clip': "Assume you are an anomaly detection expert, the object in video is a clip,Under normal circumstances, a clip can be pressed down to a certain angle and then rebound back to its original position. Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'clock': "Assume you are an anomaly detection expert, the object in video is a clock,Under normal circumstances, the second hand of a clock can rotate.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'fan': "Assume you are an anomaly detection expert, the object in video is a fan,Under normal circumstances, the fan blades can rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'gear': "Assume you are an anomaly detection expert, the object in video is a gear,Under normal circumstances, all gears rotate at a constant speed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'hinge': "Assume you are an anomaly detection expert, the object in video is a hinge,Under normal circumstances, the hinge can rotate without angle limitations and does not come loose.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'liquid': "Assume you are an anomaly detection expert, the object in video is a bottle with liquid,Under normal circumstances, the liquid is free of foreign objects and does not leak.,Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'lock': "Assume you are an anomaly detection expert, the object in video is a door lock,Under normal circumstances, when the lock is turned, the bolt can retract properly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'magnet': "Assume you are an anomaly detection expert, the object in video is a magnet,Under normal circumstances, a magnet can stick to iron.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'rolling_bearing': "Assume you are an anomaly detection expert, the object in video is rolling_bearing,Under normal circumstances, the balls inside a bearing can rotate.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'rubber_band': "Assume you are an anomaly detection expert, the object in video is rubber band,Under normal circumstances, a rubber band can stretch without cracking or breaking.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'screw': "Assume you are an anomaly detection expert, the object in video is screw,Under normal circumstances, a screw can be tightened securely.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'servo': "Assume you are an anomaly detection expert, the object in video is servo,Under normal circumstances, a servo motor can rotate without angle limitations and return to its center position.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'slide': "Assume you are an anomaly detection expert, the object in video is slide,Under normal circumstances, the sliding rail can move smoothly, and the structure remains intact.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'spherical_bearing': "Assume you are an anomaly detection expert, the object in video is spherical_bearing,Under normal circumstances, a spherical bearing can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'sticky_roller': "Assume you are an anomaly detection expert, the object in video is sticky_roller,Under normal circumstances, the roller can rotate smoothly without falling off.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'toothpaste': "Assume you are an anomaly detection expert, the object in video is toothpaste,Under normal circumstances, toothpaste does not leak when squeezed.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'usb': "Assume you are an anomaly detection expert, the object in video is usb,Under normal circumstances, the USB cap can rotate smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    'zipper': "Assume you are an anomaly detection expert, the object in video is zipper,Under normal circumstances, a zipper can open and close smoothly.Please rate the anomaly of the object in this video on a scale from 0 to 1, with 0 being definitely normal and 1 being definitely abnormal. You only need to give me an anomaly score without explanation, and the response format should be like this: {anomalyscore= },without anyother words.",
    # 添加其他类别的特定问题
}

def upload_video(image_path):
    """模拟上传视频并初始化模型和状态"""
    print(f"Uploading video from: {image_path}")
    state = default_conversation.copy()
    img_list = []
    first_run = True
    chat.upload_video(image_path, img_list)  # 使用 chat 对象上传视频
    print("Video uploaded and model initialized.")
    return state, img_list, first_run

def submit_text(state, text, img_list, first_run, temperature=0.2, max_output_tokens=5120000000):
    logger.info(f"add_text. ip:. len: {len(text)}")
    """提交文本并生成模型响应"""
    print(f"Submitting text: {text}")
    text[:1536]
    if first_run:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, img_list)
        state = default_conversation.copy()
        first_run = False
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    # state.skip_next = False 
    
    
    
    
    # 调用生成器并逐步收集响应内容
    response_generator = chat.answer(state, img_list, temperature, max_output_tokens,first_run )
    response = ""
    for res in response_generator:
    # 获取最新的对话状态
        state, chatbot_update, img_list, first_run = res[:4]
        response = state.messages[-1][-1]  # 不断更新 response 为最新生成内容

    #print("Model response:", response)  # 打印完整的模型响应
    return state, response

def clear_history():
    """清除聊天记录并重置状态"""
    print("Clearing history.")
    state = default_conversation.copy()
    img_list = []
    first_run = True
    return state, img_list, first_run

def process_and_record_video(video_path, category, chat,full_output_path, describe_output_path,explain_output_path,fifth_output_path):
    """上传视频、提交五个问题并在每次提交后记录模型响应"""
    # 上传视频并初始化状态
    state, img_list, first_run = upload_video(video_path)
    
    # 打开文件，准备写入响应
    with open(full_output_path, "a") as full_output, open(fifth_output_path, "a") as fifth_output,open(describe_output_path,"a") as describe_output,open(explain_output_path,"a") as explain_output:
        # 逐个提交问题并记录响应
        for i, question in enumerate(COMMON_QUESTIONS, 1):
            state, response = submit_text(state, question, img_list, first_run)
            full_output.write(f"{video_path}\nQ{i}: {response}\n")
            if i == len(COMMON_QUESTIONS) :
                explain_output.write(f"{video_path}\nQ{i}: {response}\n\n")
            if i == len(COMMON_QUESTIONS)-1:
                describe_output.write(f"{video_path}\nQ{i}: {response}\n\n")
                
        
        # 类别特定问题
        specific_question = CATEGORY_SPECIFIC_QUESTIONS.get(category, "Provide details for this category.")
        state, specific_response = submit_text(state, specific_question, img_list, first_run)
        
        # 写入特定问题的响应
        #full_output.write(f"{video_path}\nQ1: {specific_response}\n\n")
        fifth_output.write(f"{video_path}\nQ6: {specific_response}\n\n")
    
    # 清除历史记录
    state, img_list, first_run = clear_history()

def main_process(data_dir, output_dir, chat):
    """主函数：遍历目录结构，对每个视频处理并保存响应"""

    category=args.obj

    category_path = os.path.join(data_dir, category, "test")

    # 创建类别二级目录
    category_output_dir = os.path.join(output_dir, category)
    os.makedirs(category_output_dir, exist_ok=True)

    for defect_type in os.listdir(category_path):
        defect_path = os.path.join(category_path, defect_type)

        # 创建 defect type 级别的输出目录
        defect_output_dir = os.path.join(category_output_dir, defect_type)
        os.makedirs(defect_output_dir, exist_ok=True)

        # 设置输出文件路径（每个 defect type 单独对应一个 .txt 文件）
        full_output_path = os.path.join(defect_output_dir, "full_responses.txt")
        describe_output_path=os.path.join(defect_output_dir,"describe_respon.txt")
        explain_output_path=os.path.join(defect_output_dir,"explain_respon.txt")
        fifth_output_path = os.path.join(defect_output_dir, "score_responses.txt")

        for video_file in os.listdir(defect_path):
            video_path = os.path.join(defect_path, video_file)
            print(f"Processing video: {video_path}")

            # 处理并记录视频的响应
            process_and_record_video(video_path, category, chat, full_output_path, describe_output_path,explain_output_path,fifth_output_path)


def initialize_chat(args):
    """初始化 chat 对象"""
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)
    return chat

if __name__ == "__main__":
    # 解析参数并初始化 chat 对象
    parser = Options().initialize()
    args = parser.parse_args()
    chat = initialize_chat(args)

    # 设置数据目录路径
    data_dir = args.data_dir
    output_dir = args.output_dir
    target_categories = ["ball","button",'car','caster_wheel','clip','clock','fan','gear','hinge','liquid','lock','magnet','rolling_bearing','rubber_band','screw','servo','slide','spherical_bearing','sticky_roller','toothpaste','usb','zipper']
    main_process(data_dir, output_dir,chat)
