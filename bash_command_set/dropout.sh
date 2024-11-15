CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --dropout 0.0 --experiment_name baseline_dropout_0.0
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --dropout 0.1 --experiment_name baseline_dropout_0.1
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --dropout 0.3 --experiment_name baseline_dropout_0.3
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --dropout 0.5 --experiment_name baseline_dropout_0.5