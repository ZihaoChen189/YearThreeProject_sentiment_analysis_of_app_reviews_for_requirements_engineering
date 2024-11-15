preprocess=(content some_stop_words stem lemm all_stop_words)
for((i=0;i<${#preprocess[@]};i++))

do
    # multi-label
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-multi-label-bart-base --model_type bart-base --task multi-label --num_classes 10  --preprocess ${preprocess[$i]}
    wait
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-multi-label-roberta-base --model_type roberta-base --task multi-label --num_classes 10 --preprocess ${preprocess[$i]}
    wait
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-multi-label-distilbert-base --model_type distilbert-base-uncased --task multi-label --num_classes 10 --preprocess ${preprocess[$i]}
    wait
done
