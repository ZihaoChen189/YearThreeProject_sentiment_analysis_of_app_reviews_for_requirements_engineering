preprocess=(content some_stop_words stem lemm all_stop_words)
for((i=0;i<${#preprocess[@]};i++))

do
    # multi-class
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-single-bart-base --model_type bart-base --preprocess ${preprocess[$i]}
    wait
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-single-roberta-base --model_type roberta-base --preprocess ${preprocess[$i]}
    wait
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py --experiment_name ${preprocess[$i]}-single-distilbert-base --model_type distilbert-base-uncased  --preprocess ${preprocess[$i]}
    wait
done

