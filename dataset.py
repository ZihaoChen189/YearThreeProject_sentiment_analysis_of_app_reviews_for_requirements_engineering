import torch  # pip; for PyTorch

import numpy as np  # pip
import pandas as pd  # pip
from torch.utils.data import Dataset  # the abstract class, defined by PyTorch
from torch.utils.data import DataLoader  # batched dataset loader
# import different model tokenizer tools from the HuggingFace
# https://huggingface.co/docs/transformers/main/en/model_doc/roberta#transformers.RobertaTokenizer
# https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertTokenizer
# https://huggingface.co/docs/transformers/main/en/model_doc/bart#transformers.BartTokenizer
from transformers import RobertaTokenizer, DistilBertTokenizer, BartTokenizer


# Subclass PyTorch Dataset and override its methods to create a custom Dataset.
class BaseDataset(Dataset):
    def __init__(self, file_path, split, tokenizer_model, preprocess, max_length=512):
        self.tokenizer_model = tokenizer_model  # which model tokenizer would the system use?
        self.max_length = max_length  # the limited length of the input sequence (sentence)

        # in traditional ML approaches, the "augmentation" sheet was considered but not taken into account here
        # due to causing the possible data leakage and model over-fitting, NO reference value
        if preprocess == "augmentation":
            sheet_name = "augmentation"
            preprocess = "content"
        else:
            sheet_name = "processed"

        data = pd.read_excel(file_path, sheet_name=sheet_name)  # get the xlsx file
        manual_index = list(data["manual_index"])  # read all the manual index of app reviews
        content = list(data[preprocess])  # read all app reviews data (the input sequence/sentence)

        # a simple map, allowing the sklearn library to correctly calculate the model performance
        # mapped scores were still intuitive: 1~5 -> 0~4; -1~-5 -> 0~4
        positive_score = [abs(int(i)) - 1 for i in list(data["positive"])]
        negative_score = [abs(int(i)) - 1 for i in list(data["negative"])]

        # also, translate emotional labels into the format that machines could understand
        emotion_map = {"positive": 0, "neutral": 1, "negative": 2}
        emotion_label = [emotion_map[i] for i in list(data["emotion_label"])]  # execute the mapping operation

        with open("/".join(file_path.split("/")[:-1] + [split + ".txt"]), "r") as f:
            # retrieve indices for these different datasets (train_data, test_data, or validation_data)
            target_idx = f.read().splitlines()  # read this data
            target_idx = [int(i) for i in target_idx]  # the list format

        self.data = []  # prepare to store the data information from (train_data, test_data, or validation_data)
        for i in range(len(manual_index)):  # use a for loop to extract the manual index of ALL app reviews
            if manual_index[i] in target_idx:  # check if this manual index was same as the target index in "target_idx"
                # store that review information
                self.data.append({
                    "manual_index": manual_index[i],
                    "content": content[i],  # input sentence(sequence)
                    "positive_score": positive_score[i],  # multi-label
                    "negative_score": negative_score[i],  # multi-label
                    "emotion_label": emotion_label[i]  # multi-class
                })

        # set the model tokenizer
        if tokenizer_model.split("/")[-1].startswith("bart"):
            self.tokenizer = BartTokenizer.from_pretrained(tokenizer_model)  # Bart tokenizer
        elif tokenizer_model.split("/")[-1].startswith("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model)  # Roberta tokenizer
        elif tokenizer_model.split("/")[-1].startswith("distilbert"):
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_model)  # DistilBert tokenizer
        else:
            assert NotImplementedError

        # extract the specific index of two special tokens [UNK] and [SEP] from the vocabulary of
        # different tokenizers, to prepare for the subsequent data augmentation
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        self.sep_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

    def __len__(self):
        return len(self.data)  # the size of this dataset

    def __getitem__(self, idx):
        # get data by random idx generated by PyTorch itself
        text = self.data[idx]["content"]  # get the app review corresponding to the specific index
        # feed this review into the tokenizer, and it might be padded
        # if this review exceeded the limited max_length, it would be truncated
        tokens = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = tokens["input_ids"]  # gain a series of index values of tokens, after tokenizing that review
        input_ids = self.aug(input_ids)  # Data Augmentation
        input_ids = torch.tensor(input_ids)  # convert into the tensor format that PyTorch could understand

        # return related information
        positive_score = self.data[idx]["positive_score"]
        negative_score = self.data[idx]["negative_score"]
        emotion_label = self.data[idx]["emotion_label"]
        return input_ids, positive_score, negative_score, emotion_label


# a subclass of BaseDataset for Test / Validate, inheriting all the functionality of BaseDataset
class TestDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        assert split in ["test_data", "validation_data"]  # confirm that the parameter was passed correctly
        super(TestDataset, self).__init__(split=split, *args, **kwargs)

    def aug(self, input_ids, p=0.5):
        return input_ids


# a subclass of BaseDataset for Train, inheriting all the functionality of BaseDataset
class TrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(TrainDataset, self).__init__(split="train_data", *args, **kwargs)  # only for training

    # data augmentation during training the model
    def aug(self, input_ids, p=0.5):
        if np.random.rand() < p:  # 50% probability but with the seed in the train.py
            length = input_ids.index(self.sep_token_idx)  # determine the position of [SEP]
            for i in range(np.random.randint(1, 5)):  # how many times for data augmentation?
                # for the entire input sequence, except for [CLS], [SEP], and the ending [PAD]s, randomly replace
                # the rest of positions, by changing the original index to the index of [UNK],
                # artificially introducing noise to enhance the model's robustness, avoiding over-fitting
                input_ids[np.random.randint(1, length - 1)] = self.unk_token_idx  # position of [SEP] was used here
        return input_ids


# simple test: but the "./pretrained_models/model_name" should be prepared before running
if __name__ == "__main__":
    dataset = TrainDataset(
        file_path="./data/processed_data_after_manual.xlsx",
        tokenizer_model="./pretrained_models/roberta-base",
        preprocess="content"  # stem
    )

    # initialize an instance of TestDataset, to load the test dataset
    # dataset = TestDataset(
    #     split="test_data",  # specify this dataset split type was "test_data"
    #     file_path="./data/filtered_data_without_manual_check.xlsx",
    #     tokenizer_model="./pretrained_models/distilbert-base-uncased",  # specify the path to the tokenizer
    #     preprocess="content"
    # )

    # load the TestDataset using PyTorch's DataLoader
    # set the batch size to 32, shuffle the data, and drop the last batch if it was not full
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    for input_ids, positive_scores, negative_scores, emotion_labels in dataloader:  # iterate over the DataLoader
        # print the shapes of each batch to understand the dimensions of data
        print(input_ids.shape)  # Size([32, 512]): 32 for batch; 512 for token ids (in this file, max_length=512)
        print(positive_scores.shape)
        print(negative_scores.shape)
        print(emotion_labels.shape)
