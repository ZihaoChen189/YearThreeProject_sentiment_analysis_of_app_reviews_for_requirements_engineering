CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name bart_without_manual_check_single --model_type bart-base --data_path ./data/filtered_data_without_manual_check.xlsx
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name roberta_without_manual_check_single --model_type roberta-base --data_path ./data/filtered_data_without_manual_check.xlsx
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name distilbert_without_manual_check_single --model_type distilbert-base-uncased --data_path ./data/filtered_data_without_manual_check.xlsx
wait


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name bart_without_manual_check_multi_label --model_type bart-base --data_path ./data/filtered_data_without_manual_check_version2.xlsx --task multi-label --num_classes 10
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name roberta_without_manual_check_multi_label --model_type roberta-base --data_path ./data/filtered_data_without_manual_check_version2.xlsx --task multi-label --num_classes 10
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name distilbert_without_manual_check_multi_label --model_type distilbert-base-uncased --data_path ./data/filtered_data_without_manual_check_version2.xlsx --task multi-label --num_classes 10
