cd ..
export PYTHONPATH=$(pwd)/src/VideoLLaMA:$PYTHONPATH

python -m src.VideoLLaMA.test
python -m src.VideoLLaMA.evaluator