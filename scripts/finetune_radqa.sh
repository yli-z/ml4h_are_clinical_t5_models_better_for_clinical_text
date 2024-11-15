
MODEL_PATH="[path_to_model_directory]"
TRAIN_FILE_PATH="[path_to_train_file]"
VAL_FILE_PATH="[path_to_val_file]"
TEST_FILE_PATH="[path_to_test_file]"
OUTPUT_DIR="[path_to_output_directory]"

export PYTHONPATH=$PYTHONPATH:$(pwd)

#  Use --replace-text-with-tags for the Clinical-T5(Lehman et al., 2023) model
#  Use --is-instruction-finetuned for the FLAN-T5 model
torchrun --nproc_per_node 4 src/finetune_radqa.py \
        --model_name_or_path ${MODEL_PATH} \
        --train_file ${TRAIN_FILE_PATH} \
        --validation_file ${VAL_FILE_PATH} \
        --test_file ${TEST_FILE_PATH} \
        --context_column context \
        --question_column question \
        --answer_column answers \
        --do_train \
        --do_eval \
        --do_predict \
        --save_total_limit 1 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 30 \
        --version_2_with_negative \
        --max_seq_length 1024 \
        --load_best_model_at_end \
        --predict_with_generate \
        --metric_for_best_model "f1" \
        --report_to "wandb" \
        --evaluation_strategy "epoch" \
        --learning_rate 1e-4 \
        --lr_scheduler_type "constant" \
        --optim "adafactor" \
        --save_strategy "epoch" \
        --overwrite_output_dir \
        --seed 1 \
        --output_dir ${OUTPUT_DIR}