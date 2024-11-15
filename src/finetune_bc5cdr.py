import json
import argparse
import pandas as pd
import numpy as np 
from seqeval.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import re
import os
import logging 


logger = logging.getLogger(__name__)

def convert_BIO_labels(predictions):
    """
    convert_BIO_labels Function for BC5CDR is from:
    https://github.com/justinphan3110/SciFive/blob/main/finetune/ner/scifive_fine_tune_ner.ipynb 
    """
    result_labels = []
    cnt = 0
    for pred in predictions:
        pred = re.sub(r'\*(\w+)', r'\1*', pred)
        tokens = re.sub(r'[!"#$%&\'()+,-.:;<=>?@[\\]^_`{\|}~⁇]', ' ', pred.strip()).split()
        seq_label = []
        start_entity = 0
        entity_type = 'O'
        for idx, token in enumerate(tokens):
            if token.endswith('*'):
                start_entity += 1 if (start_entity == 0 or token[:-1] != entity_type) else -1
                entity_type = token[:-1]
            else:
                if start_entity == 0:
                    seq_label.append('O')
                    entity_type = 'O'
                elif start_entity < 0:
                    raise "Something errors"
                else:
                    if tokens[idx - 1].endswith('*'):
                        label = 'B-' + entity_type.upper()
                        seq_label.append(label)
                    else:
                        label = 'I-' + entity_type.upper()
                        seq_label.append(label)
        result_labels.append(seq_label)
        cnt += 1
    return result_labels

def compute_metrics(predictions):
    """
    Metric Function for BC5CDR is from:
    https://github.com/justinphan3110/SciFive/blob/main/finetune/ner/scifive_fine_tune_ner.ipynb 
    """
    predictions.label_ids[predictions.label_ids == -100] = 0
    
    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    pred_labels = convert_BIO_labels(decoded_predictions)
    actual_labels = convert_BIO_labels(decoded_labels)

    for i, (a, b) in enumerate(zip(pred_labels, actual_labels)):
        len_a = len(a)
        len_b = len(b)
        
        if len_a > len_b:
            pred_labels[i] = pred_labels[i][:len_b]
        elif len_a < len_b:
            pred_labels[i] = pred_labels[i] + ['PAD'] * (len_b - len_a)

    f1score = f1_score(actual_labels, pred_labels)
    recallscore = recall_score(actual_labels, pred_labels)
    precisionscore = precision_score(actual_labels, pred_labels)

    return {
            'f1': f1score, 
            'precision_score': precisionscore,
            'recall_score': recallscore
           }


def format_single_prompted_example(sentence: str) -> str:
    """Format a single example. """
    sentence = "Sentence: {} Identify and label disease terms in the sentence: ".format(sentence)

    return sentence

def preprocess_function(examples, tokenizer, max_seq_length, is_instruction_finetuned):
    """Format the examples and then tokenize them. """
    if is_instruction_finetuned:
        inputs = [format_single_prompted_example(x) for x in examples['inputs']]
    else:
        inputs = ["bc5cdr_disease_ner: " + x for x in examples['inputs']]
    targets = [x for x in examples['targets'] ]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_seq_length, truncation=True)
    return model_inputs

def get_data(train_file_path, dev_file_path, test_file_path):
    """Get the bc5cdr data. """
    dataset_dict = {}
    dataset_dict['train'] = Dataset.from_pandas(pd.read_csv(train_file_path))
    dataset_dict['val'] = Dataset.from_pandas(pd.read_csv(dev_file_path))
    dataset_dict['test'] = Dataset.from_pandas(pd.read_csv(test_file_path))
    
    return DatasetDict(dataset_dict)

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, is_instruction_finetuned):
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, is_instruction_finetuned)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
    return DatasetDict(tokenized_dict)

def train_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                args):

    if args.sample_train_percent != -1:
        tokenized_data['train'] = tokenized_data['train'].train_test_split(args.sample_train_percent)['test']


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
            lr_scheduler_type="constant",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='f1',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=args.num_beams,
            generation_max_length=args.target_seq_length,
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
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            num_train_epochs=args.num_train_epochs,
            metric_for_best_model='f1',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=args.num_beams,
            generation_max_length=args.target_seq_length,
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

    with open(output_dir + 'bc5cdr-prediction.txt', 'w+') as f:
        f.write("\n".join(predictions))
    with open(output_dir + 'bc5cdr-label.txt', 'w+') as f:
        f.write("\n".join(labels))
    with open(output_dir + 'bc5cdr-scores.json', 'w+') as f:
        json.dump(outputs.metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--train-file-path', type=str, required=True)
    parser.add_argument('--dev-file-path', type=str, required=True)
    parser.add_argument('--test-file-path', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--target-seq-length', type=int, default=512)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--sample-train-percent', type=float, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_eval', default=False, action="store_true")
    parser.add_argument('--do_predict', default=False, action="store_true")
    parser.add_argument('--num-beams', default=1, type=int)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--use-constant-adafactor', default=False, action="store_true")
    parser.add_argument('--is-instruction-finetuned', default=False, action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Running with {args.seed}")

    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.train_file_path, args.dev_file_path, args.test_file_path)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.is_instruction_finetuned)
    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)  