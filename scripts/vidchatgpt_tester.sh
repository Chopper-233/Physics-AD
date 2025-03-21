cd ..
export PYTHONPATH=$(pwd)/src/VideoChatgpt:$PYTHONPATH

python -m src.VideoChatgpt.test --obj "hinge"
python -m src.VideoChatgpt.evaluator 