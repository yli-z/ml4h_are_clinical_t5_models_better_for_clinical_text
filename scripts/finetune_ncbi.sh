
NCBI_TRAIN_FILE_PATH="[path to ncbi train file]"
NCBI_VAL_FILE_PATH="[path to ncbi val file]"
NCBI_TEST_FILE_PATH="[path to ncbi test file]"
MODEL_PATH="[path to model directory]"
OUTPUT_DIR="[path to output directory]"


#  Use --is-instruction-finetuned for the FLAN-T5 model
python src/finetune_ncbi.py \
        --ncbi-train-path ${NCBI_TRAIN_FILE_PATH} \
        --ncbi-dev-path ${NCBI_VAL_FILE_PATH} \
        --ncbi-test-path ${NCBI_TEST_FILE_PATH} \
        --model-path ${MODEL_PATH} \
        --output-dir ${OUTPUT_DIR} \
        --batch-size 16 \
        --gradient-accumulation-steps 4 \
        --lr 1e-4 \
        --num_train_epochs 30 \
        --do_train \
        --do_eval \
        --do_predict \
        --seed 1