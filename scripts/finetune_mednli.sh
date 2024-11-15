MEDNLI_TRAIN_FILE_PATH="[path to mednli train file]"
MEDNLI_VAL_FILE_PATH="[path to mednli val file]"
MEDNLI_TEST_FILE_PATH="[path to mednli test file]"
MODEL_PATH="[path to model directory]"
OUTPUT_DIR="[path to output directory]"

#  Use --replace-text-with-tags for the Clinical-T5(Lehman et al., 2023) model
#  Use --is-instruction-finetuned for the FLAN-T5 model
python src/finetune_mednli.py \
        --mednli-train-path ${MEDNLI_TRAIN_FILE_PATH}  \
        --mednli-val-path ${MEDNLI_VAL_FILE_PATH} \
        --mednli-test-path ${MEDNLI_TEST_FILE_PATH} \
        --model-path ${MODEL_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --use-constant-adafactor \
        --batch-size 64 \
        --do_train \
        --do_eval \
        --do_predict \
        --lr 1e-4 \
        --num_train_epochs 30 \
        --seed 1