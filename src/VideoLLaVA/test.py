import time
import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from dataset.vidllava_dataset import *  # Replace with your actual dataset import
from tqdm import tqdm
from options.VideoLLaVA.option import Options
import os
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

parser = Options()

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", args.local_rank))
torch.cuda.set_device(local_rank)
# os.environ['CUDA_VISIBLE_DEVICES']='6,7'

# Load the pre-trained model and processor
model_path = args.model_path
model = VideoLlavaForConditionalGeneration.from_pretrained(model_path)
processor = VideoLlavaProcessor.from_pretrained(model_path)

# Initialize the dataset and dataloader
video_dataset = Videodataset_sample(data_dir=args.data_dir)
dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=2)

# DeepSpeed inference initialization
ds_config = args.config
model = deepspeed.init_inference(
    model=model,
    config=ds_config,
)

# Open output file for writing
output_file_path = f"results/VideoLLaVA/result_{args.task}.txt"
scores = {}
gts = {}
answers = []
# with open(output_file_path, 'w') as f:
# Iterate over the dataset
for idx, (clip, video_path) in enumerate(tqdm(dataloader)):
    
    clip = clip[0]  # Get video frames
    video_path = video_path[0].replace(args.data_dir, "")
    clsname = video_path.split("/")[0]  # Extract class name
    anom_type = video_path.split("/")[1]
    # Set up prompt
    if args.task == "describ":
        prompt = "Watch this video and answer the questions: What is happening in the video? Focus on the movement of the obj and describe the content of the video in detail. Don't reply anything unnecessary."
    elif args.task == "reason":
        prompt = "Assume you are an anomaly detection expert,do you think the fuction of the object in the video is normal or abnormal, please give me a reasonable explanation."
    else:
        prompt = "Watch this video and answer the questions below: " \
            "Score the level of the possibility of the anomalies from 0 to 1, where 0 stand for 'Absolutely no abnormalities' and 1 stand for 'Absolutely abnormalities'. Try not to score precisely at 0.5. Only output the score as the answer and no other reply."

    # Prepare input for the model
    inputs = processor(
        text="USER: <video>" + prompt + " ASSISTANT:",
        videos=clip,
        return_tensors="pt"
    )
    
    # Move inputs to model device
    inputs = {k: v.to(model.module.device) for k, v in inputs.items()}
    # print(**inputs)
    # Generate model response
    generate_ids = model.generate(**inputs, max_length=150)
    answer = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Format and write answer to output
    if args.task == "auc":
        answer = answer.replace("USER: " + prompt + " ASSISTANT:", "")
        if clsname not in scores.keys():
            scores[clsname]=[float(answer.replace("USER: " + prompt + " ASSISTANT:", ""))]
        else:
            scores[clsname].append(float(answer.replace("USER: " + prompt + " ASSISTANT:","")))

        if clsname not in gts.keys():
            gts[clsname]=["anomaly_free" not in video_path]
        else:
            gts[clsname].append("anomaly_free" not in video_path)


    else:
        answer = answer.replace("USER: " + prompt + " ASSISTANT:", video_path + ":")
        answers.append(answer)



if args.task == "auc":
    with open(output_file_path, 'w') as f:
        for clsname in scores.keys():
            auc = roc_auc_score(gts[clsname], scores[clsname])
            ap = average_precision_score(gts[clsname], scores[clsname])
            acc_list = [0 if i<0.5  else 1 for i in scores[clsname]]
            acc = accuracy_score(gts[clsname], acc_list)

            print(clsname, "AUC: {:.3f}, AP: {:.3f}, ACC: {:.3f}".format(auc,ap,acc), file=f)
else:
    with open(output_file_path, 'w') as f:
        for answer in answers:
            print(answer, file=f)
