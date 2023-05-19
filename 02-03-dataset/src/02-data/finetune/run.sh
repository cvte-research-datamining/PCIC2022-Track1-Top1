i="0"

CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$i nohup python3 -u pretrain.py > log1 2>&1

CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$i nohup python3 -u finetune.py > log2 2>&1

