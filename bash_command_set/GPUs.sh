# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node 6 train.py --epochs 5 --bs 8 --experiment_name six_GPU

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --epochs 5 --bs 8 --experiment_name four_GPU

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py --epochs 5 --bs 8 --experiment_name two_GPU
