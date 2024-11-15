"""
This file is adapted from Clinical-T5(Lehman et al., 2023) repo
https://github.com/elehman16/do-we-still-need-clinical-lms/blob/main/src/finetuning/preprocess_mednli.py 
"""

import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.utils.convert_deid_tags import replace_list_of_notes
import logging 
from transformers import set_seed


logger = logging.getLogger(__name__)

def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0
    
    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    return {
            'f1': f1_score(decoded_labels, decoded_predictions, average='macro'), 
            'accuracy': accuracy_score(decoded_labels, decoded_predictions)
           }


def format_single_example(sentence1: str, sentence2: str, is_instruction_finetuned: bool) -> str:
    """Format a single example. """

    sentence1 = sentence1.encode().decode("utf-8-sig")
    sentence2 = sentence2.encode().decode("utf-8-sig")

    if is_instruction_finetuned:
        prefix = f"Answer entailment, contradiction or neutral. Premise: {sentence1} Hypothesis: {sentence2.strip()}"
    else:
        prefix = f"mednli premise: {sentence1} hypothesis: {sentence2.strip()}"
    if not(prefix[-1] == '.'):
        prefix += '.'
    return prefix

def preprocess_function(examples, tokenizer, max_seq_length: int, replace_text_with_tags: bool, is_instruction_finetuned:bool ):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(s1, s2, is_instruction_finetuned) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    targets = examples['gold_label']

    if replace_text_with_tags:
        inputs = replace_list_of_notes(inputs)

    # 1 token for the SEP between 
    to_remove = [len(tokenizer.tokenize(x)) <= (max_seq_length - 2) for x in inputs]    
    inputs = np.asarray(inputs)[to_remove].tolist()
    targets = np.asarray(targets)[to_remove].tolist()

    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_seq_length, truncation=True)
    return model_inputs

def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    return data

def get_data(mednli_train_path: str, mednli_val_path: str, mednli_test_path: str):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train_jsonl = read_jsonl(mednli_train_path)
    val_jsonl = read_jsonl(mednli_val_path)
    test_jsonl = read_jsonl(mednli_test_path)

    train = Dataset.from_list(train_jsonl)
    val = Dataset.from_list(val_jsonl)
    test = Dataset.from_list(test_jsonl)

    return DatasetDict({"train": train, "val": val, "test": test})

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, replace_text_with_tags: bool, is_instruction_finetuned: bool):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, replace_text_with_tags, is_instruction_finetuned)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)

def train_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data,
                args):


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    if args.use_constant_adafactor:
        print("Using constant Adafactor as learning rate.")
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            optim="adafactor",
            local_rank=args.local_rank,
            lr_scheduler_type="constant",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='accuracy',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=3,
            seed=args.seed,
            data_seed=args.seed,
            report_to='wandb'
        )    

    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            local_rank=args.local_rank,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='accuracy',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=3,
            seed=args.seed,
            data_seed=args.seed,
            report_to='wandb'
        )


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    outputs = trainer.predict(tokenized_data["test"])
    labels = tokenizer.batch_decode(outputs.label_ids, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)
    logger.info("*** Predict ***")
    trainer.log_metrics("predict", outputs.metrics)
    trainer.save_metrics("predict", outputs.metrics)


    with open(output_dir + 'mednli-prediction.txt', 'w+') as f:
        f.write("\n".join(predictions))
    with open(output_dir + 'mednli-label.txt', 'w+') as f:
        f.write("\n".join(labels))
    with open(output_dir + 'mednli-scores.json', 'w+') as f:
        json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mednli-dir', type=str, required=True)
    parser.add_argument('--mednli-train-path', type=str, required=True)
    parser.add_argument('--mednli-val-path', type=str, required=True)
    parser.add_argument('--mednli-test-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use-constant-adafactor', action='store_true')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--replace-text-with-tags', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_eval', default=False, action="store_true")
    parser.add_argument('--do_predict', default=False, action="store_true")
    parser.add_argument('--is-instruction-finetuned', default=False, action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Running with {args.seed}")

    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.mednli_train_path, args.mednli_val_path, args.mednli_test_path)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)


    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)

    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.replace_text_with_tags, args.is_instruction_finetuned)
    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)