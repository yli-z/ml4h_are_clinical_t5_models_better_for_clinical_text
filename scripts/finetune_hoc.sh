MODEL_PATH="[path to model directory]"
TRAIN_FILE_PATH="[path to train file]"
VAL_FILE_PATH="[path to val file]"
TEST_FILE_PATH="[path to test file]"
OUTPUT_DIR="[path to output directory]"


#  Use --is-instruction-finetuned for the FLAN-T5 model
python src/finetune_hoc.py \
            --model-path ${MODEL_PATH} \
            --output-dir ${OUTPUT_DIR} \
            --train_file ${TRAIN_FILE_PATH} \
            --dev_file ${VAL_FILE_PATH} \
            --test_file ${TEST_FILE_PATH} \
            --batch-size 64 \
            --lr 1e-4 \
            --use-constant-adafactor \
            --num_train_epochs 30 \
            --do_train \
            --do_eval \
            --do_predict \
            --seed 1