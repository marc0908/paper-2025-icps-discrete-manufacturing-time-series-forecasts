#!/bin/bash

source eval_model_paths.env

python3 eval2_recursive_lookahead.py \
  --nruns 10000 \
  --data-path datasets/pick_n_place_procedure_dataset.csv \
  --model Crossformer "$CROSSFORMER_PATH" \
  --model DUET "$DUET_PATH" \
  --model PDF "$PDF_PATH" \
  --model PatchTST "$PATCHTST_PATH" \
  --model iTransformer "$ITRANSFORMER_PATH" \
  --model "Crossformer*" "$CROSSFORMER_RETRAIN_PATH" \
  --model "DUET*" "$DUET_RETRAIN_PATH" \