import torch  # pip; for PyTorch
import numpy as np  # pip
# different kinds of classification indicators
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluator(object):
    def __init__(self, num_tasks=1) -> None:
        # num_tasks=1 meant the multi-class classifier
        # num_tasks=2 meant the multi-label classifier
        super(Evaluator, self).__init__()
        self.num_tasks = num_tasks

        # multi-class: [ [] ]; multi-label: [ [] [] ]
        self.predictions = [[] for _ in range(num_tasks)]
        self.targets = [[] for _ in range(num_tasks)]

        # corresponding labels
        if self.num_tasks == 1:
            self.labels = [["positive", "neutral", "negative"]]  # multi-class
        else:
            self.labels = [["1", "2", "3", "4", "5"], ["-1", "-2", "-3", "-4", "-5"]]  # multi-label

    # add a batch of prediction results and corresponding true labels to the evaluator
    def add_batch(self, predictions, targets, task_id=0):
        # if predictions were the Tensor data type, convert them into the CPU type that NumPy could understand
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()
        # load data at the specific position in the nested list, preparing for the evaluation
        self.predictions[task_id].append(predictions)  # APPEND
        self.targets[task_id].append(targets)  # APPEND

    def run(self):  # execute the evaluation operation
        results = []  # LIST: used for storing results

        # num_tasks=1 -> multi-class; num_tasks=2 -> multi-label
        for task_id in range(self.num_tasks):

            # flatten corresponding predictions and true labels along the axis=0 direction, to facilitate comparison
            # if task_id=1: these values would be flattened in the second nested list, within the whole list
            predictions = np.concatenate(self.predictions[task_id], axis=0)
            targets = np.concatenate(self.targets[task_id], axis=0)

            # if it was the multi-label task, TWO accuracies would be derived:
            # one for all positive scores and one for all negative scores
            acc = accuracy_score(targets, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None)  # no average

            res = {"accuracy": acc}  # add the accuracy in the dictionary for current values in the nested list
            # accuracy for the multi-class performance?
            # accuracy for the positive score performance in the multi-label?
            # accuracy for the negative score performance in the multi-label?

            # self.labels[task_id] was decided by "num_tasks", since "for task_id in range(self.num_tasks):" in line 38
            # self.labels = [["positive", "neutral", "negative"]]  # multi-class, num_tasks=1
            # self.labels = [["1", "2", "3", "4", "5"], ["-1", "-2", "-3", "-4", "-5"]]  # multi-label, num_tasks=2
            for i in range(len(self.labels[task_id])):
                # get the true label name: positive? 1? 5? -1? -5?
                label_name = self.labels[task_id][i]  # self.labels was also used here
                # record metrics WITH true label names in the dictionary
                res["precision_%s" % label_name] = precision[i]
                res["recall_%s" % label_name] = recall[i]
                res["f1-score_%s" % label_name] = f1[i]
            results.append(res)

        # whether it was the multi-class:
        # there should be 10 key-value pairs in one dictionary, in "results"
        # accuracy + precision(Positive, Neutral, Negative) + recall(PNN) + f1-score(PNN)
        if len(results) == 1:
            return results[0]  # the output of logging

        # whether it was the multi-label:
        # the list "results" would have two dictionaries, one for positive score metrics, one for negative ones
        else:
            # make it more intuitive, reset the format of the output
            res_dict = {}  # new dictionary
            acc = 0  # prepare to calculate the average of accuracy values
            for task_id in range(self.num_tasks):  # num_tasks=2 -> multi-label
                # improve the format
                for k, v in results[task_id].items():
                    # task_id = 0 meant the list with positive scores metrics;
                    # task_id = 1 meant the list with negative scores metrics;
                    # k (key) meant metrics with label names, already completed before: accuracy? precision_1? recall_1?
                    # v meant their values
                    res_dict["task_%d_" % task_id + k] = v

                acc += results[task_id]["accuracy"]  # but each dictionary contained only one accuracy value
            res_dict["accuracy"] = acc / self.num_tasks  # average
            return res_dict  # the output of logging

    # reset predictions and targets to prepare for further evaluations
    # delete previous objects and set up new ones
    def reset(self):
        del self.predictions
        del self.targets
        self.predictions = [[] for _ in range(self.num_tasks)]
        self.targets = [[] for _ in range(self.num_tasks)]
