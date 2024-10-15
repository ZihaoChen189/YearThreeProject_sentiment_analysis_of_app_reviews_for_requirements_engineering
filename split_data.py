import pandas as pd  # pip
import random

data = pd.read_excel("processed_data_after_manual.xlsx")  # read the xlsx file
# retrieve all the manual index of reviews and convert it into the list format
all_manual_index = list(data["manual_index"])

random.seed(68)  # set the random seed for experiment reproducibility
random.shuffle(all_manual_index)  # randomly shuffle this index list based on the fixed seed

train_number = int(len(all_manual_index) * 0.8)  # how much was the train data, based on the given proportion
test_number = int(len(all_manual_index) * 0.15)  # how much was the test data, based on the given proportion

# create these txt files, storing the manual index of reviews for training data, test data, and validation data,
# based on the 80%,  15%, and  5% ratio

# use the for loop to extract each index sequentially from the "all_manual_index" according to the specific proportion,
# and write them into corresponding files, separated by "\n"

with open("train_data.txt", "w") as f:
    f.write("\n".join(str(i) for i in all_manual_index[:train_number]))

with open("test_data.txt", "w") as f:
    f.write("\n".join(str(i) for i in all_manual_index[train_number:train_number + test_number]))

with open("validation_data.txt", "w") as f:
    f.write("\n".join(str(i) for i in all_manual_index[train_number + test_number:]))
