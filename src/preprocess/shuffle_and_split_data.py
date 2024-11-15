import json
import pandas as pd
import numpy as np 
import random
import os


def shuffle_and_split_mednli(train_jsonl, val_jsonl, test_jsonl, output_folder_path, num_fold=5):

    train_premise_values = set(map(lambda x:x["sentence1"], train_jsonl))
    val_premise_values = set(map(lambda x:x["sentence1"], val_jsonl))

    union_premise_values = train_premise_values.union(val_premise_values)
    
    union_list = list(union_premise_values)

    random.seed(42)
    random.shuffle(union_list)

    chunks = np.array_split(union_list, num_fold)
    train_fold_dict = {}
    val_fold_dict = {}

    for i in range(num_fold):
        val_data = []
        train_data = []

        for example in (train_jsonl + val_jsonl):
            if example["sentence1"] in chunks[i]:
                val_data.append(example)
            else:
                train_data.append(example)
        
        if not os.path.exists(f"{output_folder_path}/fold_{i}"):
            os.makedirs(f"{output_folder_path}/fold_{i}")

        train_fold_dict[i] = train_data
        val_fold_dict[i] = val_data

        with open(f"{output_folder_path}/fold_{i}/train_{i}.jsonl", "w+") as f:
            for example in train_data:
                json.dump(example, f)
                f.write('\n')
        
        with open(f"{output_folder_path}/fold_{i}/val_{i}.jsonl", "w+") as f:
            for example in val_data:
                json.dump(example, f)
                f.write('\n')
        print(f"[ cv-{i} done]")

    # Check if the data is correctly split
    for i in range(num_fold):
        val_fold_premise = set(map(lambda x:x["sentence1"], val_fold_dict[i]))
        train_fold_premise = set(map(lambda x:x["sentence1"], train_fold_dict[i]))
        assert(val_fold_premise.intersection(train_fold_premise) == set())
        for j in range(num_fold):
            if i != j:
                j_fold_premise = set(map(lambda x:x["sentence1"], train_fold_dict[j]))
                assert(val_fold_premise.intersection(j_fold_premise) == val_fold_premise)
    

def shuffle_and_split(train_path: str, val_path: str, save_folder_path, is_tsv=False, num_fold=5):
    """
    Shuffle and split the data into num_fold folds.
    You may modify this function depending on data formats.
    """
    train_df = pd.read_csv(train_path) if not is_tsv else pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path) if not is_tsv else pd.read_csv(val_path, sep="\t")

    merged_train_val = pd.concat([train_df, val_df])
    merged_train_val = merged_train_val.sample(frac=1).reset_index(drop=True)

    splitted_list = np.array_split(merged_train_val, num_fold)

    # split the data into num_fold folds
    for fold_index in range(num_fold):
        fold_train_df = pd.concat([splitted_list[i] for i in range(num_fold) if i != fold_index])
        fold_val_df = splitted_list[fold_index]
        fold_train_df.to_csv(f"{save_folder_path}/cv_{fold_index}_train.csv", index=False)
        fold_val_df.to_csv(f"{save_folder_path}/cv_{fold_index}_val.csv", index=False)
        