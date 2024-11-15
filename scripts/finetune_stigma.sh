MODEL_PATH="[path to model directory]"
OUTPUT_DIR="[path to output directory]"
DATA_PATH="[path to data directory]"


#  Use --is-instruction-finetuned for the FLAN-T5 model
for KEYWORD in "adamant" "compliance" "other"
done
    for FOLD_INDEX in 0 1 2 3 4
    do
        python src/finetune_stigma.py \
            --model-path $MODEL_PATH \
            --output-dir $OUTPUT_DIR \
            --keyword $KEYWORD \
            --data-path $DATA_PATH \
            --batch-size 4 \
            --gradient-accumulation-steps 16 \
            --use-constant-adafactor \
            --num_train_epochs 30 \
            --lr 1e-4 \
            --do_train \
            --do_eval \
            --do_predict \
            --cross-validation \
            --fold-index $FOLD_INDEX \
            --seed 1
    done
done