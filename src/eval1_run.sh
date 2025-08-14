#!/bin/bash

source eval_model_paths.env

python3 eval1_single_loookahead.py \
  --nruns 100 \
  --data-path datasets/pick_n_place_procedure_dataset.csv \
  --model Crossformer "$CROSSFORMER_PATH" \
  --model DLinear "$DLINEAR_PATH" \
  --model DUET "$DUET_PATH" \
  --model PDF "$PDF_PATH" \
  --model PatchTST "$PATCHTST_PATH" \
  --model iTransformer "$ITRANSFORMER_PATH"
