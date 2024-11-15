# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name 4GPU_3090 --epochs 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name 4GPU_V100 --epochs 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name 4GPU_A4000 --epochs 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name 4GPU_3080 --epochs 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name 4GPU_3060 --epochs 5