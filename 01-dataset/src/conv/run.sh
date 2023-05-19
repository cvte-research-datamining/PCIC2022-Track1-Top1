i="0"

CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$i nohup python3 -u conv.py > log 2>&1
