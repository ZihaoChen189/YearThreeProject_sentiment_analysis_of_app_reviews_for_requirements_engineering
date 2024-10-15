import torch.nn as nn  # import PyTorch neural network module for defining network layers
import torch.nn.functional as F  # activation functions

# import different models from the transformers library for loading pre-trained models
from transformers import RobertaModel, DistilBertModel, BartModel


class BertClassifier(nn.Module):
    def __init__(self, model_name, output_size, hidden_size=768, dropout=0.0):
        super(BertClassifier, self).__init__()
        if model_name.split("/")[-1].startswith("bart"):
            # select the target pre-train encoder-decoder model from the "pretrained_models" folder
            self.encoder = BartModel.from_pretrained(model_name)

            # extract "sequence of hidden states at the output of the last layer of the decoder of the model"
            # https://huggingface.co/docs/transformers/main/en/model_doc/bart#transformers.BartModel
            # Also, Figure 3a from https://doi.org/10.48550/arXiv.1910.13461
            self.pooling_function = self.pooling_fn_cls_token

            # for p in self.encoder.shared.parameters():
            #     p.requires_grad = False  # freeze all weights of the encoder embedding layers to prevent over-fitting
            # # freeze the weights of the first three layers of the encoder model to prevent over-fitting
            # for m in self.encoder.encoder.layers[:3]:
            #     for p in m.parameters():
            #         p.requires_grad = False

        elif model_name.split("/")[-1].startswith("roberta"):
            # select the target pre-train encoder model from the "pretrained_models" folder
            self.encoder = RobertaModel.from_pretrained(model_name)

            # extract "last layer hidden state of the first token of the sequence (classification token)"
            # https://huggingface.co/docs/transformers/main/en/model_doc/roberta#transformers.RobertaModel
            self.pooling_function = self.pooling_fn_roberta

            # for p in self.encoder.embeddings.parameters():
            #     p.requires_grad = False  # freeze all weights of the encoder embedding layers to prevent over-fitting
            # # freeze the weights of the first six layers of the encoder model to prevent over-fitting
            # for m in self.encoder.encoder.layer[:6]:
            #     for p in m.parameters():
            #         p.requires_grad = False

        elif model_name.split("/")[-1].startswith("distilbert"):
            # select the target pre-train encoder model from the "pretrained_models" folder
            self.encoder = DistilBertModel.from_pretrained(model_name)

            # extract "sequence of hidden-states at the output of the last layer of the model"
            # https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertModel
            self.pooling_function = self.pooling_fn_cls_token

            # for p in self.encoder.embeddings.parameters():
            #     p.requires_grad = False  # freeze all weights of the encoder embedding layers to prevent over-fitting
            # # freeze the weights of the first three layers of the encoder model to prevent over-fitting
            # for m in self.encoder.transformer.layer[:3]:
            #     for p in m.parameters():
            #         p.requires_grad = False

        else:
            assert NotImplementedError

        # the shape of the encoder/decoder output might be (batch_size, 768)
        # the linear layer received the output from the encoder/decoder, with the fixed 768 feature length
        self.fc1 = nn.Linear(768, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalisation Layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # output layer
        self.dropout = nn.Dropout(dropout)  # Dropout Layer

    def pooling_fn_cls_token(self, outputs):
        # [:, 0]: could "imagine" the cube with (batch_size, sequence_length, hidden_size)
        # extract all the "batch_size" and the first position of the "sequence_length", which was [CLS]
        return outputs[0][:, 0]  # the first position of the model output

    def pooling_fn_roberta(self, outputs):
        return outputs[1]  # the second position of the roberta output, deriving the "exclusive" classification token

    def forward(self, x):
        # the typical forward operation of the network
        # x might be Tensor(4, 512)
        # emb might be Tensor(4, 768)
        # output might be from Tensor(4, 768) to Tensor(4, 3) for multi-class
        outputs = self.encoder(x.long())  # PyTorch standard: torch.int64 before passing through the encoder
        emb = self.pooling_function(outputs)  # apply pooling functions to encoder/decoder outputs to obtain embeddings
        emb = self.dropout(emb)  # apply dropout to prevent over-fitting
        out = self.fc1(emb)  # pass embeddings through the first fully connected layer
        out = self.bn1(out)  # apply batch normalization to improve stability and speed up convergence
        out = F.relu(out)  # activation function to introduce non-linearity
        # out = F.gelu(out)
        out = self.fc2(out)  # pass the result through the second fully connected layer

        return out  # return the final output of the network


# simple test: but the "./pretrained_models/model_name" should be prepared before running
if __name__ == "__main__":
    from dataset import TestDataset  # import the pre-defined TestDataset class from the dataset module
    from torch.utils.data import DataLoader  # import DataLoader from PyTorch

    # initialize an instance of TestDataset, to load the validation dataset
    dataset = TestDataset(
        split="validation_data",  # specify this dataset split type was "validation_data"
        file_path="./data/processed_data_after_manual.xlsx",
        tokenizer_model="./pretrained_models/distilbert-base-uncased",  # specify the path to the tokenizer
        preprocess="content"
    )

    # load the TestDataset class using PyTorch's DataLoader
    # set the batch size to 4, shuffle the data, and drop the last batch if it was not full
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    # initialize the model for multi-class
    model = BertClassifier('./pretrained_models/distilbert-base-uncased', 3, 768)
    # model = BertClassifier('./pretrained_models/distilbert-base-uncased', 10, 768)  # this was for multi-label

    # iterate through the DataLoader, processing each batch
    for input_ids, pos_scores, neg_scores, emotion_labels in dataloader:
        outputs = model(input_ids)  # feed input index data to the model, to get the output
        print(outputs.shape)  # dimensions of the model output
