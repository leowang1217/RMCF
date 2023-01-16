model_dir="checkpoints/rmcf"
data_path="geom-drugs"

args=(
    --train-prefix  $data_path/train
    --valid-prefix  $data_path/valid
    --test-prefix   $data_path/test
    --model-dir $model_dir
    --seg-vocab-path $data_path/hit.pkl
    --vocab-path $data_path/vocab.pkl
    --model-name rmcf
    --mpnn-steps 3
    --max-steps 1200000
    --batch-size 256
    --learning-rate 5e-4
    --logging-steps 500
    --eval-steps 20000
    --save-steps 20000
    --save-total-limit 60
    --num-workers 4
    --update-freq 1
    --fp16
)
function dist-train(){
        python3 -m torch.distributed.launch \
            --nproc_per_node=8 \
            train.py \
            "$@"
}

dist-train "${args[@]}" >> $model_dir/log.txt