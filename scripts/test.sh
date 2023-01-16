model_dir="checkpoints/rmcf/checkpoint-20000"
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
    --batch-size 256
)

python3 test.py "${args[@]}" >> $model_dir/test_log.txt