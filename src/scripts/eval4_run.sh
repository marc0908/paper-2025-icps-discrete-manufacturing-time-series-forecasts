#!/bin/bash

source eval_model_paths.env

python3 eval4_lookahead_w_override.py \
  --nruns 146 \
  --data-path datasets/pick_n_place_procedure_w_overrides.csv \
  --model Crossformer "$CROSSFORMER_PATH" \
  --model DUET "$DUET_PATH" \
  --model PDF "$PDF_PATH" \
  --model DLinear "$DLINEAR_PATH" \
  --model PatchTST "$PATCHTST_PATH" \
  --model iTransformer "$ITRANSFORMER_PATH" \
  --model "Crossformer*" "$CROSSFORMER_RETRAIN_PATH" \
  --model "DUET*" "$DUET_RETRAIN_PATH" \
