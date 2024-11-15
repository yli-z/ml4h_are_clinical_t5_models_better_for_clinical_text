"""
This file is adapted from Clinical-T5(Lehman et al., 2023) repo
https://github.com/elehman16/do-we-still-need-clinical-lms/blob/main/src/finetuning/preprocess_clip.py
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.utils.convert_deid_tags import replace_list_of_notes
import logging
from transformers import set_seed

logger = logging.getLogger(__name__)


LABEL_TYPES = ['Appointment-related followup',
               'Medication-related followups',   
               'Other helpful contextual information',   
               'Lab-related followup',   
               'Case-specific instructions for patient',   
               'Procedure-related followup',   
               'Imaging-related followup']

LOWERCASED_TYPES = [x.lower() for x in LABEL_TYPES]

def str_tags_to_binary_tensor(los):
    """Convert list of strings to indexed positions. """
    arr = np.zeros(len(LABEL_TYPES))
    for str_ in los:
        if not str_ in LABEL_TYPES and not str_ in LOWERCASED_TYPES: continue
        
        # It's in our list. Get the label. Mark as 1 in our label list.
        if str_ in LOWERCASED_TYPES: arr[LOWERCASED_TYPES.index(str_)] = 1 
        else: arr[LABEL_TYPES.index(str_)] = 1

    return arr

def get_span_indicies(offset_mapping, st_ch_index, end_ch_index):
    """Get the start and end token indicies given character offsets. """
    start_tk_index, end_tk_index = -1, float('inf')
    for i, offset in enumerate(offset_mapping):
        st, end = offset[0], offset[1]
        if st > st_ch_index and start_tk_index == -1:
            start_tk_index = i - 1
        
        if st >= end_ch_index:
            end_tk_index = i
            break
    
    end_tk_index = min(end_tk_index, len(offset_mapping))
    assert(start_tk_index != -1)

    assert(start_tk_index != -1)
    return start_tk_index, end_tk_index 

def load_ids(clip_train_id_path, clip_val_id_path, clip_test_id_path):
    """Load the training/val/test ids. """
    tr_id = pd.read_csv(clip_train_id_path, header=None)
    vl_id = pd.read_csv(clip_val_id_path, header=None)
    te_id = pd.read_csv(clip_test_id_path, header=None)

    return set(tr_id[0].values), set(vl_id[0].values), set(te_id[0].values)

def load_data(clip_path) -> pd.DataFrame:
    """Load the data from the sentences.csv. """
    df = pd.read_csv(clip_path + '/sentence_level.csv')

    df['sentence'] = [eval(x) for x in df['sentence']]
    df['labels'] = [eval(x) for x in df['labels']]

    # Combine the text, remember sentence offsets. 
    docs = {'note_id': [], 'text': [], 'labels': []}
    for id, group in df.groupby('doc_id'):
        text, sentence_offsets = "", []
        
        for _, row in group.iterrows():
            sent = ' '.join(row['sentence'])
    
            # Remove the weird `I-` in front of the labels
            row['labels'] = [x[2:] for x in row['labels']]
            row['labels'].sort()
            labels = ', '.join(row['labels'])
            sentence_offsets.append((len(text), len(text) + len(sent), labels))
            # Now join the text
            text += sent + ' '

        docs['note_id'].append(id)
        docs['text'].append(text)
        docs['labels'].append(sentence_offsets)

    data_df = pd.DataFrame(docs)
    return data_df

def split_into_chunks(tokenizer, df: pd.DataFrame, replace_text_with_tags: bool, max_seq_length: int, is_instruction_finetuned: bool) -> pd.DataFrame:
    """Split the data into chunks of `max_seq_length`.  Attempt to take an equal number
    of tokens before and after the text.
    @param tokenizer 
    @param replace_text_with_tags
    @param max_seq_length """
    inputs = []
    count = 1
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        tokenized_text = tokenizer(row.text, return_offsets_mapping=True, add_special_tokens=True)

        # Iterate through the sentences for this note
        for sentence in row.labels:
            
            st_token, end_token = get_span_indicies(tokenized_text['offset_mapping'], sentence[0], sentence[1])

            # If we can't fit it in the context length, just truncate...
            if end_token - st_token + 2 > max_seq_length / 2:
                end_token = st_token + int(max_seq_length / 2 - 2)

            # Take a chunk of text (2 extra tokens for </s> and the ";" delimiter.
            context_size = int((max_seq_length - (2 * (end_token - st_token) + 2)) / 2)
            start_chunk = max(st_token - context_size, 0)
            end_chunk = min(end_token + context_size, len(tokenized_text['input_ids']))

            # If we are hitting the end or at the start, then just expand our starting chunk
            if st_token - start_chunk != context_size:
                end_chunk = end_token + (context_size * 2 - (st_token - start_chunk)) 

            elif end_chunk - end_token != context_size:
                start_chunk = st_token - (context_size * 2 - (end_chunk - end_token))

            # Select the chunk of the text AND the actual sentence we care about
            # A chunk is the paragraph that contains the sentence, including before and after sentences
            span = tokenized_text['offset_mapping'][st_token:end_token]
            selected_offsets = tokenized_text['offset_mapping'][start_chunk:end_chunk]
            selected_text = row.text[span[0][0]:span[-1][1]]    
            chunk_text = row.text[selected_offsets[0][0]:selected_offsets[-1][1]]

            # Concatenate the text delimited by a ;
            text = chunk_text + ' ; ' + selected_text

            # add prefix
            text = "clip: " + text
            # If we want to modify the text, do it now.
            if replace_text_with_tags:
                text = replace_list_of_notes([text])[0] 

            if is_instruction_finetuned:
                text = "Context: {context}. Label the above sentence as an empty string or as one or more of the following options, delimited by comma: Options: {options}".format(context=text, options=(", ").join(LABEL_TYPES))

            if count < 10:
                count += 1

            # Tokenize the text
            if is_instruction_finetuned:
                # Set max_seq_length * 2 because we are concatenating both prompts and text
                model_inputs = tokenizer(text, text_target=sentence[2], max_length=max_seq_length*2, truncation=True)
            else:
                model_inputs = tokenizer(text, text_target=sentence[2], max_length=max_seq_length + 2, truncation=True)
        
            # Add to our lists
            model_inputs['id'] = row.note_id 
            inputs.append(model_inputs)
         
            # Some quick assertions to make sure we aren't going crazy! 
            assert(end_token - st_token < max_seq_length - 2)
            assert(end_chunk - start_chunk + 2 + (end_token - st_token) == max_seq_length)
    
    return pd.DataFrame(inputs)


def get_data(tokenizer, clip_path, clip_train_id_path, clip_val_id_path, clip_test_id_path, replace_text_with_tags, max_seq_length, is_instruction_finetuned ) -> DatasetDict:
    """Get the CLIP data. 
    @param tokenizer is a Huggingface tokenizer.
    @param clip_path is the path to the clip data.
    @param replace_text_with_tags determines whether or not we modify the text to remove PHI.
    @param max_seq_length is the maximum sequence length."""
    df = load_data(clip_path)
    tr_id, vl_id, te_id = load_ids(clip_train_id_path, clip_val_id_path, clip_test_id_path) 
    
    # Split the data into chunks + into train/val/test
    model_inputs = split_into_chunks(tokenizer, df, replace_text_with_tags, max_seq_length,is_instruction_finetuned)
    train = model_inputs[model_inputs['id'].isin(tr_id)]
    val = model_inputs[model_inputs['id'].isin(vl_id)]
    test = model_inputs[model_inputs['id'].isin(te_id)]

    # Create the dataset objects and return 
    input_dict = {'train': Dataset.from_pandas(train), 'val': Dataset.from_pandas(val), 'test': Dataset.from_pandas(test)}  

    return DatasetDict(input_dict)


def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0

    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    
    index_labels = [str_tags_to_binary_tensor(l.split(', ')) for l in decoded_labels]
    index_preds = [str_tags_to_binary_tensor(p.split(', ')) for p in decoded_predictions]
    
    return {
            'f1': f1_score(index_labels, index_preds, average='macro'),
            'accuracy': accuracy_score(index_labels, index_preds),
            'micro_f1': f1_score(index_labels, index_preds, average='micro'),
            'auc_score': roc_auc_score(index_labels, index_preds)
        }


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
            generation_num_beams=5,
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
            generation_num_beams=5,
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

    with open(output_dir + '/clip-prediction.txt', 'w+') as f:
        f.write("\n".join(predictions))
    with open(output_dir + '/clip-label.txt', 'w+') as f:
        f.write("\n".join(labels))
    with open(output_dir + '/clip-scores.json', 'w+') as f:
        json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="What is the model to load?")
    parser.add_argument('--clip-dir', type=str, required=True, help="Where is the CLIP data stored?")
    parser.add_argument('--clip-train-id-path', type=str, required=True, help="Where is the train id stored?")
    parser.add_argument('--clip-val-id-path', type=str, required=True, help="Where is the val id stored?")
    parser.add_argument('--clip-test-id-path', type=str, required=True, help="Where is the test id stored?")
    parser.add_argument('--lr', type=float, default=1e-4, help="")
    parser.add_argument('--replace-text-with-tags', action='store_true', help="Should we replace PHI w/ tags?")
    parser.add_argument('--max-seq-length', type=int, default=256, help="What is the maximum sequence length to consider?")
    parser.add_argument('--output-dir', type=str, required=True, help="Where should we store the model?")
    parser.add_argument('--seed', type=int, default=1, help="What seed to run with?")
    parser.add_argument('--local_rank', type=int, default=-1) 
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--sample-train-percent', default=-1, type=float)
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_eval', default=False, action="store_true")
    parser.add_argument('--do_predict', default=False, action="store_true")
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--use-constant-adafactor', default=False, action="store_true")
    parser.add_argument('--is-instruction-finetuned', default=False, action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)

    # Load the data and train the model
    model_inputs = get_data(tokenizer, args.clip_dir, args.clip_train_id_path, args.clip_val_id_path, args.clip_test_id_path, args.replace_text_with_tags, args.max_seq_length, args.is_instruction_finetuned)
    
    train_model(tokenizer=tokenizer, model=model, output_dir=args.output_dir, tokenized_data=model_inputs, args=args)