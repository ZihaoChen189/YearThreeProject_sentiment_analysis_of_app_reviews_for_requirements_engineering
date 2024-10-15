import os
import argparse
import logging
import random

import numpy as np  # pip
import torch  # pip; for PyTorch
import torch.nn as nn  # import PyTorch neural network module for defining network layers
import torch.optim as optim  # pip for PyTorch optimization function
from torch.utils.data import DataLoader  # batched dataset loader
from torch.utils.tensorboard import SummaryWriter  # tensorboard
from tqdm import tqdm  # pip for the progress bar

# two classes for multiple GPU training: "nn.DataParallel" and "nn.parallel.DistributedDataParallel"
# PyTorch encouraged the latter one: "create a process for each GPU, avoiding the potential performance overhead"
# https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel
from torch.nn.parallel import DistributedDataParallel  # better distributed data processing using multiple GPUs
# if "DistributedDataParallel" was selected, using "torch.distributed.launch" to launch the program
# in this project, the system environment could be viewed as the "multi-processing" training because of many GPUs,
# but it should be "singe node" training not multi node because only one machine terminal was used

from dataset import TrainDataset, TestDataset  # pre-defined different Dataset processors for train, validate and test
from model import BertClassifier  # pre-defined Classifier
from metrics import Evaluator  # pre-defined Evaluator for evaluating the model performance


# set a series of random seeds for possible random operations that might be called, during the training process,
# thereby ensuring a certain level of the experimental reproducibility
def set_seed(seed):
    random.seed(seed)  # python seed
    np.random.seed(seed)  # numpy seed
    torch.manual_seed(seed)  # seed for generating random numbers for pytorch objects
    torch.cuda.manual_seed(seed)  # generating random numbers on the current GPU
    torch.cuda.manual_seed_all(seed)  # generating random numbers on all GPUs
    # use "cuDNN" to select the fastest convolution algorithm at the backend level of the pytorch,
    # improving the whole system speed; however, it would affect the experiment reproducibility to some extent

    # based on the specific model architecture, it would search for optimization algorithms during runtime
    torch.backends.cudnn.benchmark = True


# ensure GPUs didn't read the same data between each other
# and set the seed to maintain a certain level of reproducibility
# passed as an input parameter to the Dataloader
# https://pytorch.org/docs/stable/notes/randomness.html#dataloader
def worker_init_fn_seed(worker_id):
    seed = torch.initial_seed()
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    # each worker for "base_seed + worker_id" for "Randomness in multi-process data loading"
    seed = (seed + worker_id) % (2 ** 32)
    np.random.seed(seed)


# start training :)
def train(args):
    # create the specific experiment folder to store result files
    args.save_dir = os.path.join(args.save_dir, args.experiment_name)
    # the location of pre-download pretrained models
    args.model_type = os.path.join("pretrained_models", args.model_type)

    # every GPU would start a new process
    # here, only the first GPU was responsible for the corresponding terminal output and logging record,
    # managing the use of GPUs and avoiding possible confusion
    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)  # check the result file location
        writer = SummaryWriter(args.save_dir)  # tensorboard object to record performance

        logger = logging.getLogger()  # initial logging object
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(os.path.join(args.save_dir, "train.log"), mode="w")  # one log file for each experiment
        fh.setLevel(logging.INFO)  # only keep main information in this log file

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # output all information on the terminal for debugging and observation

        logger.addHandler(ch)  # reset the logging direction
        logger.addHandler(fh)  # reset the logging direction

    # model initialization
    model = BertClassifier(args.model_type, args.num_classes, args.hidden_size, args.dropout)
    model.cuda()

    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#distributeddataparallel
    # perform the gradient synchronization on this "model", with each GPU obtaining different gradients
    # take the average of all GPU gradients without manual operation
    model = DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss()  # cross-entropy loss function for classification problems

    # three different Dataset objects for training, validating, and testing
    train_set = TrainDataset(args.data_path, tokenizer_model=args.model_type, preprocess=args.preprocess,
                             max_length=args.max_length)
    val_set = TestDataset("validation_data", args.data_path, tokenizer_model=args.model_type,
                          preprocess=args.preprocess, max_length=args.max_length)
    test_set = TestDataset("test_data", args.data_path, tokenizer_model=args.model_type, preprocess=args.preprocess,
                           max_length=args.max_length)

    # set up subset samplers to prevent the same data from being used across GPUs
    # to maintain the different data order in each epoch, MUST be created before "train_loader" (or DataLoader)
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    # three different DataLoader objects for train, validate, and test
    # "num_workers" meant each process(GPU) created multiple subprocesses to read data, which would be faster,
    # but mainly depended on how many cores the CPU had
    # "pin_memory" accelerated the speed of transferring data to the GPU
    train_loader = DataLoader(train_set, batch_size=args.bs, num_workers=4, sampler=train_sampler, pin_memory=True,
                              worker_init_fn=worker_init_fn_seed)
    val_loader = DataLoader(val_set, batch_size=args.bs, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, pin_memory=True)

    # filter parameters that only needed gradient updates; otherwise it might result in errors
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)  # less sensitive to the learning rate
    evaluator = Evaluator(1 if args.task == "single" else 2)  # 1 for multi-class; 2 for multi-label
    best_val_acc = 0.0  # record the best performance on the val_set
    best_test_metrics = None  # model performance on the test_set
    total_step = 0  # one step was one time of updating gradients(forward()+backword()) -> "optimizer.step()"

    for epoch in range(args.epochs):  # 20 fixed epochs
        model.train()  # training pattern: important for Dropout and Norm layers

        # train_sampler.set_epoch(epoch) MUST be utilized before the "train_loader" initialization,
        # otherwise "the same ordering will always be used"
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        train_sampler.set_epoch(epoch)  # maintain the different data order in each epoch

        if args.local_rank == 0:  # the GPU management, avoiding possible confusion
            iterator = tqdm(train_loader)
        else:
            iterator = train_loader

        for inputs, pos_scores, neg_scores, emo_labels in iterator:
            # get the final output results: [_, 3] for multi-class or [_, 10] for multi-label
            outputs = model(inputs.cuda())

            if args.task == "single":
                emo_labels = emo_labels.cuda()  # true values
                loss = criterion(outputs, emo_labels).mean()  # mean-loss for one whole batch
            elif args.task == "multi-label":
                pos_scores = pos_scores.cuda()  # true values
                neg_scores = neg_scores.cuda()  # true values
                # the final output results had 10 feature dimensions, split them apart,
                # 5 for positive scores, and the latter 5 for negative scores
                outputs = outputs.chunk(2, dim=1)
                # mean-loss for both positive scores and negative scores
                # actually, one criterion had already considered the mean() operation
                loss = criterion(outputs[0], pos_scores) + criterion(outputs[1], neg_scores)

            # before back-propagation, reset the previously accumulated gradients to zero, manually
            # if gradients accumulated, the results became inaccurate since the direction of gradients was affected
            optimizer.zero_grad()
            loss.backward()  # back-propagation for gradients with "requires_grad=True"
            optimizer.step()  # update

            if args.local_rank == 0 and total_step % 10 == 0:  # the GPU management, avoiding possible confusion
                writer.add_scalar("train_loss", loss.item(), total_step)  # record loss in the tensorboard per 10 steps
            total_step += 1  # step++

        # start validation
        if args.local_rank == 0:  # the GPU management, avoiding possible confusion
            # test() function was defined below
            eval_metrics_val = test(model, val_loader, evaluator, args.task)  # val_loader for val_set
            eval_metrics_test = test(model, test_loader, evaluator, args.task)  # test_loader for test_set

            # record logging information
            logger.info(f"Epoch {epoch + 1}/{args.epochs}:")
            logger.info("Validation Results: " + "  ".join(["%s: %.4f" % (k, v) for k, v in eval_metrics_val.items()]))
            logger.info("Test Results: " + "  ".join(["%s: %.4f" % (k, v) for k, v in eval_metrics_test.items()]))
            for k, v in eval_metrics_val.items():
                writer.add_scalar("val/%s" % k, v, total_step)  # record model performance in the tensorboard
            for k, v in eval_metrics_test.items():
                writer.add_scalar("test/%s" % k, v, total_step)  # record model performance in the tensorboard

            val_acc = eval_metrics_val["accuracy"] 
            if val_acc > best_val_acc:  # if the current model had higher validation accuracy
                best_val_acc = val_acc  # update
                best_test_metrics = eval_metrics_test  # record the corresponding test results
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_weights.pth"))  # save this pth file
            logger.info("\n")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_weights.pth"))  # save the current one

    if args.local_rank == 0:  # the GPU management, avoiding possible confusion
        logger.info("Final Test Results: " + "  ".join(["%s: %.4f" % (k, v) for k, v in best_test_metrics.items()]))


@torch.no_grad()  # disable gradient calculation, and proceed directly with inference for acceleration
def test(model, loader, evaluator, task):
    evaluator.reset()  # reset the previous Evaluator
    model.eval()  # not model.train()
    for inputs, pos_scores, neg_scores, emo_labels in loader:
        # similar to training, but directly compare predicted values with true values, without loss consideration
        outputs = model(inputs.cuda())
        if task == "single":  # multi-class
            emo_labels = emo_labels  # true values
            preds = outputs.argmax(1).data.cpu()  # predicted values, might be (batch, 3) -> (batch,)
            # add them to the Evaluator, containing the prepared list, for calculating model performance
            evaluator.add_batch(preds, emo_labels)
        elif task == "multi-label":  # multi-label
            outputs = model(inputs.cuda())
            # the final output results had 10 feature dimensions, split them apart,
            # 5 for positive scores, and the latter 5 for negative scores
            outputs = outputs.chunk(2, dim=1)
            pos_preds = outputs[0].argmax(1)  # catch predicted values for positive scores, (batch, 5)->(batch,)
            neg_preds = outputs[1].argmax(1)  # catch predicted values for negative scores, (batch, 5)->(batch,)
            # add them to the Evaluator containing the prepared lists, for calculating model performance
            evaluator.add_batch(pos_preds, pos_scores, 0)
            evaluator.add_batch(neg_preds, neg_scores, 1)

    eval_metrics = evaluator.run()  # calculate the model performance using sklearn api
    return eval_metrics  # dictionary output to the terminal and .log file


def main():
    parser = argparse.ArgumentParser()  # parse some parameters passed in, from the command line

    # configure specific command line arguments #
    parser.add_argument("--seed", type=int, default=68, help="random seed")
    parser.add_argument("--preprocess", type=str, default="content", choices=["augmentation", "content",
                        "some_stop_words", "stem", "lemm", "all_stop_words"])
    parser.add_argument("--hidden_size", type=int, default=768, help="embedding dimensions of hidden layers")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16, help="batch size per gpu")
    parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=1e-8, help="weight decay")  # 1e-5 -> 1e-8
    parser.add_argument("--dropout", type=float, default=0.0)  # 0.1 -> 0.0
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--max_length", type=int, default=256, help="max length of tokens")  # 512 -> 256
    parser.add_argument("--model_type", type=str, choices=['bart-base', 'roberta-base', 'distilbert-base-uncased'],
                        default='roberta-base')
    parser.add_argument("--data_path", type=str, default="./data/processed_data_after_manual.xlsx")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument("--task", type=str, choices=["single", "multi-label"], default="single")
    parser.add_argument("--local-rank", type=int)

    args = parser.parse_args()  # get the namespace of all command line arguments, just like a python object

    torch.cuda.set_device(args.local_rank)  # the system automatically allocated these GPUs

    # "nccl" backend was suggested by Pytorch
    # https://pytorch.org/docs/stable/distributed.html#backends
    torch.distributed.init_process_group(backend="nccl")

    # set the random seed before experiments started, ensuring a certain level of the experimental reproducibility
    set_seed(args.seed)

    train(args)  # execute the experiment :)


if __name__ == "__main__":
    main()  # cannot run as a simple test without the "--local-rank" default value, different from previous python files
