
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-8 --max_length 256  --experiment_name baseline_256_length

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-8 --max_length 512  --experiment_name baseline_512_length
