import datetime
import os
import time
import json
from video_chatgpt.video_conversation import default_conversation
from video_chatgpt.utils import build_logger, violates_moderation, moderation_msg
from video_chatgpt.demo.chat import Chat
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *

# 初始化 logger 和模型
logger = build_logger("script_logger", "script.log")
model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model("/home/lc/Desktop/wza/cxt/Video-ChatGPT/LLaVA-7B-Lightening-v1-1", "/home/lc/Desktop/wza/cxt/Video-ChatGPT/Video-ChatGPT-7B/video_chatgpt-7B.bin")
replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
chat = Chat("/home/lc/Desktop/wza/cxt/Video-ChatGPT/LLaVA-7B-Lightening-v1-1", "video-chatgpt_v1", tokenizer, image_processor, vision_tower, model,replace_token)

# 获取对话日志文件名
def get_conv_log_filename():
    t = datetime.datetime.now()
    log_dir = "logs"  # 设置日志目录
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")

# 保存投票
def vote_last_response(state, vote_type):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state.dict()
        }
        logger.info(f"Vote saved: {vote_type}")
        fout.write(json.dumps(data) + "\n")

# 定义按钮的逻辑
def upvote_last_response(state):
    logger.info("Upvoted.")
    vote_last_response(state, "upvote")

def downvote_last_response(state):
    logger.info("Downvoted.")
    vote_last_response(state, "downvote")

def flag_last_response(state):
    logger.info("Flagged.")
    vote_last_response(state, "flag")

def regenerate(state):
    logger.info("Regenerating response.")
    state.messages[-1][-1] = None  # 清空上一个消息的内容
    state.skip_next = False
    return chat.answer(state)

def clear_history():
    logger.info("Clearing history.")
    return default_conversation.copy()

# 添加文本或视频内容到对话中
def add_text(state, text, first_run=False):
    logger.info(f"Adding text: {text[:50]}")
    if len(text) <= 0:
        state.skip_next = True
        return None
    
    # Moderation check
    flagged = violates_moderation(text)
    if flagged:
        state.skip_next = True
        return moderation_msg

    text = text[:1536]
    if first_run:
        text = text[:1200]
        text += "\n<video>" if '<video>' not in text else ''
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False

    return chat.answer(state)

# 批量上传和处理视频的逻辑
def upload_and_process_videos(video_paths, questions):
    for video_path in video_paths:
        with open(video_path, "rb") as video:
            # 上传视频
            img_list = []
            state = default_conversation.copy()
            llm_message = chat.upload_video(video, img_list)
            logger.info(f"Uploaded video: {video_path}")

            # 处理每个问题
            for question in questions:
                response = add_text(state, question, first_run=True)
                print(f"Response to '{question}' for '{video_path}': {response}")

# 使用示例
video_paths = ["/home/lc/Desktop/wza/cxt/Video-ChatGPT/video_chatgpt/demo/demo_sample_videos/sample_2.mp4", "video_chatgpt/demo/demo_sample_videos/sample_6.mp4"]
questions = ["Why is this video strange?", "Describe the main event.", "What unusual objects are visible?"]
upload_and_process_videos(video_paths, questions)
