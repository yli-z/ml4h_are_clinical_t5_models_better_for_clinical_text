MEDNLI_TRAIN_FILE_PATH="[path_to_mednli_train_file]"
MEDNLI_VAL_FILE_PATH="[path_to_mednli_val_file]"
MEDNLI_TEST_FILE_PATH="[path_to_mednli_test_file]"
MODEL_PATH="[path_to_model_directory]"
OUTPUT_DIR="[path_to_output_directory]"

export PYTHONPATH=$PYTHONPATH:$(pwd)

#  Use --replace-text-with-tags for the Clinical-T5(Lehman et al., 2023) model
#  Use --is-instruction-finetuned for the FLAN-T5 model
python src/finetune_mednli.py \
        --mednli-train-path $MEDNLI_TRAIN_FILE_PATH  \
        --mednli-val-path $MEDNLI_VAL_FILE_PATH \
        --mednli-test-path $MEDNLI_TEST_FILE_PATH \
        --model-path $MODEL_PATH \
        --output-dir $OUTPUT_DIR \
        --use-constant-adafactor \
        --batch-size 64 \
        --do_train \
        --do_eval \
        --do_predict \
        --lr 1e-4 \
        --num_train_epochs 30 \
        --seed 1