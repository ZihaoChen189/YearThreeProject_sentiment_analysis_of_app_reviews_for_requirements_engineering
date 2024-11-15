# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name bart_model_time --model_type bart-base

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name roberta_model_time --model_type roberta-base

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name distilbert_model_time --model_type distilbert-base-uncased
