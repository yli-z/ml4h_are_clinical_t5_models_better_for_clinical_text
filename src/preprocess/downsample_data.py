import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
import json
import argparse
import pdb
import random
import os
import datasets
from datasets import load_dataset, concatenate_datasets


def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path, mode="r", encoding='utf-8-sig') as f:
        data = [json.loads(line) for line in f]
    return data

def downsample_mednli_data(mednli_train_path: str, fold: int):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train_jsonl = read_jsonl(mednli_train_path)

    # Premise-Hypothesis pairs are mostly 1 to 3 mapping, i.e. one preimise with 3 hypothesis (entailment, neutral, contradiction) in the MedNLI dataset. 
    # So we shuffle and downsample the data based on unique premises. 
    train_premise = set(map(lambda x:x["sentence1"], train_jsonl))  

    # sample 1% premise
    train_premise_list = list(train_premise)
    train_premise_list = sorted(train_premise_list)
    random.seed(42)
    random.shuffle(train_premise_list)
    train_premise_list = train_premise_list[:len(train_premise_list) // 100]

    train_df = pd.DataFrame()
    for example in train_jsonl:
        if example["sentence1"] in train_premise_list:
            train_df = train_df.append(example, ignore_index=True)

    mednli_dir = os.path.dirname(mednli_train_path)

    train_df.to_json(f"{mednli_dir}/downsampled_train_{fold}.jsonl", orient="records", lines=True)
    print("[ Downsampled MedNLI data (fold-{}) saved ]".format(fold))

    downsampled_train_jsonl = read_jsonl(f"{mednli_dir}/downsampled_train_{fold}.jsonl")
    assert(all(i in train_jsonl for i in downsampled_train_jsonl))

def shuffle_and_downsample_stigma_data(full_data_path, downsampled_save_path, percentage_list):

    data = pd.read_csv(full_data_path)
    
    shuffled_data = data.sample(frac=1)
    shuffled_data.to_csv(downsampled_save_path+"/shuffled_data.csv", index=False)
    print("[shuffled stigma data saved to {}]".format(downsampled_save_path+"/shuffled_data.csv"))

    # Downsample training data by keyword category
    for percentage in percentage_list:
        train_data_list = []
        for keyword in ["adamant", "compliance", "other"]:
            # Downsample training data by cv fold
            for cv_idx in range(5):
                fold_train_data =  shuffled_data[(shuffled_data['keyword_category'] == keyword) & (shuffled_data['split'] == 'train') & (shuffled_data['fold_eval'] == cv_idx)]
                downsampled_fold_train_data = fold_train_data.head(round(len(fold_train_data)*(percentage/100)))
                train_data_list.append(downsampled_fold_train_data)

        # Concatenate downsampled training data with original dev and test data
        downsampled_data = pd.concat(train_data_list + [data[data['split']!='train']])

        # Save downsampled data
        downsampled_data.to_csv(downsampled_save_path+f"/{percentage}%_data.csv", index=False)
        print(f"[{percentage}% downsampled stigma data saved to {downsampled_save_path}/{percentage}%_data.csv]")