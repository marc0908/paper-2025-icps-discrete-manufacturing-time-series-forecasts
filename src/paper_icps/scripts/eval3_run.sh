#!/bin/bash

source eval_model_paths.env

python eval3_train_2nd_pass.py \
    --data-path datasets/pick_n_place_procedure_dataset.csv \
    --data-path-overrides datasets/pick_n_place_procedure_w_overrides.csv \
    --model-path "$CROSSFORMER_PATH" \
    --num-epochs 100 \
    --reduction-factor 3 \
    --patience 10

python eval3_train_2nd_pass.py \
    --data-path datasets/pick_n_place_procedure_dataset.csv \
    --data-path-overrides datasets/pick_n_place_procedure_w_overrides.csv \
    --model-path "$DUET_PATH" \
    --num-epochs 100 \
    --reduction-factor 3 \
    --patience 10