export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export epochs=200
export num_workers=8

python Xu_tsne.py

