import json
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import logging 
from hoc_metrics import hoc_metric


logger = logging.getLogger(__name__)

classes = ['evading growth suppressors', 
'tumor promoting inflammation', 
'enabling replicative immortality', 
'cellular energetics', 
'resisting cell death', 
'activating invasion and metastasis', 
'genomic instability and mutation', 
'', 
'inducing angiogenesis', 
'sustaining proliferative signaling', 
'avoiding immune destruction']

prompt_classes = ['evading growth suppressors', 
'tumor promoting inflammation', 
'enabling replicative immortality', 
'cellular energetics', 
'resisting cell death', 
'activating invasion and metastasis', 
'genomic instability and mutation', 
'inducing angiogenesis', 
'sustaining proliferative signaling', 
'avoiding immune destruction']


def get_label(target_labels):
    labels = []
    for label_idx in target_labels.split('|'):
        labels.append(classes[int(label_idx)])
    return ", ".join(labels)

def preprocess_function(examples, tokenizer, max_seq_length):
    """Format the examples and then tokenize them. """
    inputs = examples['inputs']
    model_inputs = tokenizer(inputs, text_target=examples['targets'], max_length=max_seq_length, truncation=True)
    return model_inputs

def format_hoc_dataset(dataset, input_col, target_col, is_instruction_finetuned: bool):
    inputs = []
    targets = []
    for example in dataset:
        if is_instruction_finetuned:
            input_text = "Sentence: {} Assign the above sentence as zero or more of the following class labels: {}.".format(example[input_col], ", ".join(prompt_classes))
        else:
            input_text = 'hoc: ' + example[input_col]
        inputs.append(input_text)
        targets.append(get_label(example[target_col]))

    return Dataset.from_dict({'inputs': inputs, 'targets': targets})

def get_data(hoc_train_path, hoc_dev_path, hoc_test_path, is_instruction_finetuned:bool):
    """Get the hoc data. """
    dataset_dict = {}

    dataset_dict['train'] = format_hoc_dataset(Dataset.from_pandas(pd.read_csv(hoc_train_path)), 'inputs', 'targets', is_instruction_finetuned)
    dataset_dict['val'] = format_hoc_dataset(Dataset.from_pandas(pd.read_csv(hoc_dev_path)), 'inputs', 'targets', is_instruction_finetuned)
    dataset_dict['test'] = format_hoc_dataset(Dataset.from_pandas(pd.read_csv(hoc_test_path)), 'inputs', 'targets', is_instruction_finetuned)
    
    return DatasetDict(dataset_dict)

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
    return DatasetDict(tokenized_dict)

def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0

    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    
    return hoc_metric(decoded_labels, decoded_predictions)
    

def train_model(model,
                tokenizer,
                output_dir,
                tokenized_data,
                args):

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
    )

    if args.use_constant_adafactor:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            local_rank=args.local_rank,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            optim="adafactor",
            lr_scheduler_type="constant",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='micro_f1',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_max_length=args.target_seq_length,
            seed=args.seed,
            data_seed=args.seed,
            report_to='wandb',
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            do_predict=args.do_predict,
            local_rank=args.local_rank,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='micro_f1',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_max_length=args.target_seq_length,
            seed=args.seed,
            data_seed=args.seed,
            report_to='wandb',
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
    
    with open(output_dir + 'hoc-prediction.txt', 'w+') as f:
        f.write("\n".join(predictions))
    with open(output_dir + 'hoc-label.txt', 'w+') as f:
        f.write("\n".join(labels))
    with open(output_dir + 'hoc-scores.json', 'w+') as f:
        json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="What is the model to load?")
    parser.add_argument('--lr', type=float, default=1e-4, help="")
    parser.add_argument('--max-seq-length', type=int, default=256, help="What is the maximum sequence length to consider?")
    parser.add_argument('--target-seq-length', type=int, default=64)
    parser.add_argument('--output-dir', type=str, required=True, help="Where should we store the model?")
    parser.add_argument('--seed', type=int, default=42, help="What seed to run with?")
    parser.add_argument('--local_rank', type=int, default=-1) 
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--sample-train-percent', default=-1, type=float)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_eval', default=False, action="store_true")
    parser.add_argument('--do_predict', default=False, action="store_true")
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--use-constant-adafactor', default=False, action="store_true")
    parser.add_argument('--is-instruction-finetuned', default=False, action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_dict = get_data(args.train_file, args.dev_file, args.test_file, args.is_instruction_finetuned)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length)
    train_model(tokenizer=tokenizer, model=model, output_dir=args.output_dir, tokenized_data=tokenized_datasets, args=args)