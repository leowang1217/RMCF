model_dir="checkpoints/rmcf/checkpoint-20000" #your model path
data_path="geom-drugs"

args=(
    
    --test-prefix   $data_path/test
    --model-dir $model_dir
    --seg-vocab-path $data_path/hit.pkl
    --vocab-path $data_path/vocab.pkl
    --sampling-strategy random
    --cov-thres 1.25
    --model-name rmcf
    --mpnn-steps 3
    --batch-size 16
)


python3 test.py "${args[@]}"