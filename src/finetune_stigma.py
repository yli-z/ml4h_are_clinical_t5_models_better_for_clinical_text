import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)
import logging


logger = logging.getLogger(__name__)


INSTRUCTIONS = {
    "adamant": "Classify this sentence as difficult, disbelief, or exclude, regarding the credibility and obstinacy of the patient: ",
    "compliance": "Classify this sentence as negative, neutral, or positive, regarding the patient's compliance with medical advice: ",
    "other": "Classify this sentence as exclude, negative, neutral, or positive, regarding the patient's behavior and demeanor: ",
}

def compute_metrics(predictions):
    """Given some predictions, calculate the F1."""
    predictions.label_ids[predictions.label_ids == -100] = 0

    # Decode the predictions + labels
    decoded_labels = tokenizer.batch_decode(
        predictions.label_ids, skip_special_tokens=True
    )
    decoded_predictions = tokenizer.batch_decode(
        predictions.predictions, skip_special_tokens=True
    )

    return {
        "f1": f1_score(decoded_labels, decoded_predictions, average="macro"),
        "micro_f1": f1_score(decoded_labels, decoded_predictions, average="micro"),
        "precision": precision_score(
            decoded_labels, decoded_predictions, average="macro"
        ),
        "micro_precision": precision_score(
            decoded_labels, decoded_predictions, average="micro"
        ),
        "recall": recall_score(decoded_labels, decoded_predictions, average="macro"),
        "micro_recall": recall_score(
            decoded_labels, decoded_predictions, average="micro"
        ),
        "accuracy": accuracy_score(decoded_labels, decoded_predictions),
    }

def preprocess_function(
    examples,
    tokenizer,
    max_seq_length: int,
    is_instruction_finetuned: bool,
    keyword: str,
):
    """Format the examples and then tokenize them."""

    if is_instruction_finetuned:
        inputs = add_instructions(examples, keyword)["note_text"]
    else:
        # add prefix to the notes
        prefix = keyword + ": "
        inputs = prefix + examples["note_text"]
    
    inputs = inputs.tolist()

    targets = examples["label"].tolist()
    if is_instruction_finetuned:
        model_inputs = tokenizer(
            inputs, text_target=targets, padding="max_length", truncation=False
        )
    else:
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_seq_length, truncation=True
        )
    return model_inputs


def add_instructions(notes, keyword):
    """Add instructions to notes."""

    notes["note_text"] = INSTRUCTIONS[keyword] + notes["note_text"]
    return notes


def get_data(
    data_path: str,
    keyword: str,
    cross_validation: bool,
    fold_index: int
):
    """Get the stigma data."""

    data_df = pd.read_csv(data_path)
    # select the data based on the keyword
    filtered_data = data_df[data_df["keyword_category"] == keyword]
    if cross_validation:
        # Split the data into train, val, test
        train = filtered_data.loc[
            (
                (filtered_data["split"] == "train")
                & (filtered_data["fold_eval"] != fold_index)
            )
        ]
        val = filtered_data.loc[
            (
                (filtered_data["split"] == "dev")
                & (filtered_data["fold_eval"] == fold_index)
            )
        ]
        test = filtered_data.loc[filtered_data["split"] == "test"]
        test = add_instructions(test, keyword)
        return DatasetDict({"train": train, "val": val, "test": test})
    else:
        # Split the data into train, val, test
        train = filtered_data[filtered_data["split"] == "train"]
        val = filtered_data[filtered_data["split"] == "dev"]
        test = filtered_data[filtered_data["split"] == "test"]
        test = add_instructions(test, keyword)
        return DatasetDict({"train": train, "val": val, "test": test})


def get_tokenized_data(
    dataset_dict: DatasetDict,
    tokenizer,
    max_seq_length: int,
    is_instruction_finetuned: bool,
    keyword: str,
):
    """Tokenize stuff."""
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(
            dataset_,
            tokenizer,
            max_seq_length,
            is_instruction_finetuned,
            keyword=keyword,
        )
        tokenized_dict[name] = Dataset.from_dict(processed_item)
    return DatasetDict(tokenized_dict)


def save_outputs(output_dir, predictions, labels, metrics, cv_idx=None):
    if cv_idx is not None:
        with open(output_dir + "/cv-" + str(cv_idx) + "/stigma-prediction.txt", "w+") as f:
            f.write("\n".join([str(pred) for pred in predictions]))
        with open(output_dir + "/cv-" + str(cv_idx) + "/stigma-label.txt", "w+") as f:
            f.write("\n".join([str(label) for label in labels]))
        with open(output_dir + "/cv-" + str(cv_idx) + "/stigma-scores.json", "w+") as f:
            json.dump(metrics, f)
    else:
        with open(output_dir + "stigma-prediction.txt", "w+") as f:
            f.write("\n".join(predictions))
        with open(output_dir + "stigma-label.txt", "w+") as f:
            f.write("\n".join(labels))
        with open(output_dir + "stigma-scores.json", "w+") as f:
            json.dump(metrics, f)


def train_model(model, tokenizer, output_dir: str, tokenized_data, args):

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    cv_idx = args.fold_index if args.cross_validation else None

    if args.use_constant_adafactor:
        print("Using constant Adafactor as learning rate.")
        training_args = Seq2SeqTrainingArguments(
            output_dir=(
                output_dir if cv_idx is None else (output_dir + "/cv-" + str(cv_idx))
            ),
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
            metric_for_best_model="f1",
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=args.num_beams,
            seed=args.seed,
            data_seed=args.seed,
            report_to="wandb",
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=(
                output_dir if cv_idx is None else (output_dir + "/cv-" + str(cv_idx))
            ),
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
            metric_for_best_model="f1",
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=args.num_beams,
            seed=args.seed,
            data_seed=args.seed,
            report_to="wandb",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    outputs = trainer.predict(tokenized_data["test"])
    labels = tokenizer.batch_decode(outputs.label_ids, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)
    logger.info("*** Predict ***")
    trainer.log_metrics("predict", outputs.metrics)
    trainer.save_metrics("predict", outputs.metrics)

    save_outputs(output_dir, predictions, labels, outputs.metrics, cv_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--keyword", type=str, choices=["adamant", "compliance", "other"], required=True
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use-constant-adafactor", default=False, action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_predict", default=False, action="store_true")
    parser.add_argument("--cross-validation", default=False, action="store_true")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-beams", default=3, type=int)
    parser.add_argument(
        "--is-instruction-finetuned", default=False, action="store_true"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Running with {args.seed}")

    if (
        args.cross_validation and args.fold_index is None
    ):  # Number of folds is not defined
        raise ValueError("--fold-index is required for cross validation")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, from_flax=True
        )

    # Get data and use the tokenizer on the data
    dataset_dict = get_data(
        args.data_path,
        args.keyword,
        args.cross_validation,
        args.fold_index,
    )

    tokenized_datasets = get_tokenized_data(
        dataset_dict,
        tokenizer,
        args.max_seq_length,
        args.is_instruction_finetuned,
        args.keyword,
    )

    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)