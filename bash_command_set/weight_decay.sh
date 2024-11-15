CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-3 --dropout 0.0 --experiment_name baseline_wd_1e-3
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-4 --dropout 0.0 --experiment_name baseline_wd_1e-4
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-5 --dropout 0.0 --experiment_name baseline_wd_1e-5
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-6 --dropout 0.0 --experiment_name baseline_wd_1e-6
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-7 --dropout 0.0 --experiment_name baseline_wd_1e-7
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --wd 1e-8 --dropout 0.0 --experiment_name baseline_wd_1e-8

