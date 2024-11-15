CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 1e-5 --experiment_name baseline_lr_1e-5
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 2e-5 --experiment_name baseline_lr_2e-5
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 3e-5 --experiment_name baseline_lr_3e-5
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 9e-5 --experiment_name baseline_lr_9e-5
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 1e-4 --experiment_name baseline_lr_1e-4
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 3e-4 --experiment_name baseline_lr_3e-4
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --lr 9e-4 --experiment_name baseline_lr_9e-4
