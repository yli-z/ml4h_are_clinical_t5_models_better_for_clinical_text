
CLIP_DIR="[path_to_clip_data_directory]"
CLIP_TRAIN_ID_PATH="[path_to_clip_train_id_file]"
CLIP_VAL_ID_PATH="[path_to_clip_val_id_file]"
CLIP_TEST_ID_PATH="[path_to_clip_test_id_file]"
MODEL_PATH="[path_to_model_directory]"
OUTPUT_DIR="[path_to_output_directory]"

export PYTHONPATH=$PYTHONPATH:$(pwd)

#  Use --replace-text-with-tags for the Clinical-T5(Lehman et al., 2023) model
#  Use --is-instruction-finetuned for the FLAN-T5 model
torchrun --nproc_per_node 4 src/finetune_clip.py \
            --clip-dir $CLIP_DIR \
            --clip-train-id-path $CLIP_TRAIN_ID_PATH \
            --clip-val-id-path $CLIP_VAL_ID_PATH \
            --clip-test-id-path $CLIP_TEST_ID_PATH \
            --model-path $MODEL_PATH \
            --output-dir $OUTPUT_DIR \
            --lr 1e-4 \
            --batch-size 16 \
            --use-constant-adafactor \
            --num_train_epochs 30 \
            --do_train \
            --do_eval \
            --do_predict \
            --seed 1