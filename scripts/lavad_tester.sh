#!/bin/bash
# 00
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
dataset_parent_dir="/home/lc/Desktop/wza/gy/phys"
obj="ball"
dataset_dir="${dataset_parent_dir}/${obj}/test"
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------

echo "ensure you have extracted frames first"
cd ..
echo "step 1 done"
# 01
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

batch_size=32
frame_interval=1
# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

# Define pretrained model names array
pretrained_model_names=("pretrained-weights/huggingface/hub/blip2-opt-6.7b-coco"
    # /home/lc/Desktop/wza/gy/dyna/lavad-main/libs/huggingface/hub/models--Salesforce--blip2-opt-6.7b-coco
    # /home/lc/Desktop/wza/gy/dyna/lavad-main/libs/huggingface/hub/models--Salesforce--blip2-opt-6.7b-coco/snapshots/0d580de59320a25a4d2c386387bcef310d5f286e
    # "Salesforce/blip2-opt-6.7b"
    # "Salesforce/blip2-flan-t5-xxl"
    # "Salesforce/blip2-flan-t5-xl"
    # "Salesforce/blip2-flan-t5-xl-coco"
)
HUGGINGFACE_HUB_CACHE="pretrained-weights/huggingface/hub/"

for pretrained_model_name in "${pretrained_model_names[@]}"; do
    echo "Processing model: $pretrained_model_name"

    output_dir="${dataset_dir}/captions/raw/${pretrained_model_name}/"

    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python -m models.LAVAD.image_captioner \
        --root_path "$root_path" \
        --annotationfile_path "$annotationfile_path" \
        --batch_size "$batch_size" \
        --frame_interval "$frame_interval" \
        --pretrained_model_name "$pretrained_model_name" \
        --output_dir "$output_dir"
done
echo "step 2 done"
# 02
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

batch_size=32
frame_interval=4
index_dim=1024

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

cap_model_names=(
    "$dataset_dir/captions/raw/libs/huggingface/hub/blip2-opt-6.7b-coco/"
    # "$dataset_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
    # "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
    # "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
    # "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
)

cap_model_names_str=$(IFS=' '; echo "${cap_model_names[*]}")

# Extract names and concatenate with "+"
names=""
IFS='/' read -ra components <<< "$cap_model_names_str"
for component in "${components[@]}"; do
    if [[ "$component" =~ ^blip2- ]]; then
        names+="${component#blip2-}+"
    fi
done

# Remove the trailing "+" if present
names=${names%+}

echo "Creating index for $names"

index_name="index_flat_ip"
output_dir="${dataset_dir}/index/${names}/${index_name}/"
# shellcheck disable=SC2086 # We want to pass a list of strings

python -m models.LAVAD.create_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "${output_dir}" \
    --captions_dirs $cap_model_names_str
echo "step 3 done"
# 03
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

batch_size=32
frame_interval=4
fps=60  # Change this to the frame rate of your videos
clip_duration=10
num_samples=10
num_neighbors=1
index_name="opt-6.7b-coco"  # Change this to the index name you created in scripts/02_create_index.sh

echo "Processing index: $index_name"

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

captions_dir_template="$dataset_dir/captions/raw/libs/huggingface/hub/{}/"
index_dir="$dataset_dir/index/${index_name}/index_flat_ip/"
output_dir="${dataset_dir}/captions/clean/$index_name/"
python -m models.LAVAD.image_text_caption_cleaner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "$output_dir" \
    --captions_dir_template "${captions_dir_template}" \
    --index_dir "${index_dir}" \
    --fps "$fps" \
    --clip_duration "$clip_duration" \
    --num_samples "$num_samples" \
    --num_neighbors "$num_neighbors"
echo "step 4 done"
# 04
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=4
index_name="opt-6.7b-coco"  # Change this to the index name you created in scripts/02_create_index.sh

echo "Processing index: $index_name"

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

captions_dir="$dataset_dir/captions/clean/$index_name/"

# Generate a 6-digit timestamp based on the current time
exp_id=$(date +%s | tail -c 7)

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

output_scores_dir="${dataset_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${dataset_dir}/captions/summary/${llm_model_name}/$index_name/"

torchrun \
    --nproc_per_node 2 --nnodes 1 -m models.LAVAD.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --summary_prompt "$summary_prompt" \
    --output_summary_dir "$output_summary_dir" \
    --captions_dir "$captions_dir" \
    --ckpt_dir pretrained-weights/llama-2-13b-chat/ \
    --tokenizer_path pretrained-weights/tokenizer.model

torchrun \
    --nproc_per_node 2 --nnodes 1 -m models.LAVAD.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_summary_dir "$output_summary_dir" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --ckpt_dir pretrained-weights/llama-2-13b-chat/ \
    --tokenizer_path pretrained-weights/tokenizer.model \
    --score_summary
echo "step 5 done"
# # 05
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------

llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=4
index_dim=1024
index_name="opt-6.7b-coco"  # Change this to the index name you created in scripts/02_create_index.sh

# exp_id = 695600

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

captions_dir="${dataset_dir}/captions/summary/${llm_model_name}/${index_name}/"
output_dir="${dataset_dir}/index/summary/${llm_model_name}/${index_name}/index_flat_ip/"

python -m models.LAVAD.create_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
echo "step 6 done"
# 06
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=4
fps=60  # Change this to the frame rate of your videos
T=10
N=10
num_neighbors=10

# exp_id="735180" # Change this to the experiment ID from scripts/04_query_llm.sh
index_name="opt-6.7b-coco"  # Change this to the index name you created in scripts/02_create_index.sh

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

captions_dir="$dataset_dir/captions/summary/${llm_model_name}/$index_name/"
index_dir="$dataset_dir/index/summary/${llm_model_name}/$index_name/index_flat_ip/"
scores_dir="$dataset_dir/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_scores_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${dataset_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
output_similarity_dir="${dataset_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_filenames_dir="${dataset_dir}/filenames/clean_summary/${llm_model_name}/${index_name}/"

# Run the Python script with the specified parameters
python -m models.LAVAD.video_text_score_refiner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_scores_dir "$output_scores_dir" \
    --output_summary_dir "$output_summary_dir" \
    --output_similarity_dir "$output_similarity_dir" \
    --output_filenames_dir "$output_filenames_dir" \
    --captions_dir "$captions_dir" \
    --index_dir "$index_dir" \
    --scores_dir "$scores_dir" \
    --fps "$fps" \
    --clip_duration "$T" \
    --num_samples "$N" \
    --num_neighbors "$num_neighbors"
echo "step 7 done"
# 07
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

frame_interval=4
num_neighbors=10
video_fps=60  # Change this to the frame rate of your videos

# exp_id="339014"  # Change this to the experiment ID from scripts/04_query_llm.sh
index_name="opt-6.7b-coco"  # Change this to the index name you created in scripts/02_create_index.sh

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

captions_dir="${dataset_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
scores_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
similarity_dir="${dataset_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.LAVAD.eval \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --scores_dir "$scores_dir" \
    --similarity_dir "$similarity_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --frame_interval "$frame_interval" \
    --num_neighbors "$num_neighbors" \
    --without_labels \
    --video_fps "$video_fps"
    # --visualize \
    
echo "step 8 done"
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
python -m src.LAVAD.evaluator \
    --root_path "$dataset_parent_dir" \
    --obj "$obj" 

echo "step 9 done"
