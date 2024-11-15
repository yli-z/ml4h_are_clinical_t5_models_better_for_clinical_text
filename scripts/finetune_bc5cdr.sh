TRAIN_FILE_PATH="[path_to_train_file]"
VAL_FILE_PATH="[path_to_val_file]"
TEST_FILE_PATH="[path_to_test_file]"
MODEL_PATH="[path_to_model_directory]"
OUTPUT_DIR="[path_to_output_directory]"

export PYTHONPATH=$PYTHONPATH:$(pwd)

#  Use --is-instruction-finetuned for the FLAN-T5 model
python src/finetune_bc5cdr.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR \
    --train-file-path $TRAIN_FILE_PATH \
    --dev-file-path $VAL_FILE_PATH \
    --test-file-path $TEST_FILE_PATH \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --lr 1e-4 \
    --num_train_epochs 30 \
    --use-constant-adafactor \
    --do_train \
    --do_eval \
    --do_predict \
    --seed 1