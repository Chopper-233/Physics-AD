cd ..
export PYTHONPATH=$(pwd)/src/VideoLLaMA:$PYTHONPATH

python -m src.VideoLLaMA.test --obj "hinge"
python -m src.VideoLLaMA.evaluator