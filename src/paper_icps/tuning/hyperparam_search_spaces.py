# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
import copy
from typing import Dict, NoReturn

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle

#tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))

#sys.path.append(tfb_path)
#os.environ["PYTHONPATH"] = tfb_path
#print(sys.path)

from paper_icps.ts_benchmark.models.model_loader import get_models
from paper_icps.ts_benchmark.data import data_source
from paper_icps.ts_benchmark.baselines.duet.models.duet_model import DUETModel
import time
from paper_icps.torch.utils.data import Dataset, DataLoader


import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
import tempfile


from ray import train
from ray.train import Checkpoint, CheckpointConfig

import gc

import paper_icps.core.common as common
import paper_icps.core.config as config


def assemble_setup(setup_name):
    setups = {
        "duet": (
            config.default_eval_config(),
            config.model_config(
                "duet.DUET", decoder_input_required=False, has_loss_importance=True
            ),
            duet_searchspace(),
        ),
        "crossformer": (
            config.default_eval_config(),
            config.model_config(
                "time_series_library.Crossformer", "transformer_adapter"
            ),
            crossformer_searchspace(),
        ),
        "itransformer": (
            config.default_eval_config(),
            config.model_config(
                "time_series_library.iTransformer", "transformer_adapter"
            ),
            itransformer_searchspace(),
        ),
        "timexer": (
            config.default_eval_config(),
            config.model_config(
                "timexer.TimeXer", decoder_input_required=True, has_loss_importance=False
            ),
            timexer_searchspace(),
        ),
        "dlinear": (
            config.default_eval_config(),
            config.model_config("time_series_library.DLinear", "transformer_adapter"),
            dlinear_searchspace(),
        ),
        "patchtst": (
            config.default_eval_config(),
            config.model_config("time_series_library.PatchTST", "transformer_adapter"),
            patchtst_searchspace(),
        ),
        "pdf": (
            config.default_eval_config(),
            config.model_config("pdf.PDF", decoder_input_required=False),
            pdf_searchspace(),
        )
    }

    return setups[setup_name]


def duet_searchspace():
    search_space = {
        "CI": 1,  # channel independent
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "dropout": tune.uniform(0.001, 0.2),
        "e_layers": tune.choice([1, 2, 3]),
        "factor": tune.randint(1, 20),
        "fc_dropout": tune.uniform(0.0001, 0.2),
        "horizon": 400,
        "k": tune.choice([1, 2, 3]),
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "n_heads": tune.randint(2, 20),
        "norm": True,
        "num_epochs": config.max_epochs,
        "num_experts": tune.randint(2, 20),
        "patch_len": tune.choice([10, 50, 100, 200, 400]),
        "patience": 10,
        "moving_avg": tune.choice([1, 3, 5]),
        "seq_len": 1600,
    }
    return search_space


def crossformer_searchspace():
    search_space = {
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "dropout": tune.uniform(0.0001, 0.2),
        "e_layers": tune.choice([1, 2, 3]),
        "horizon": 400,
        "n_heads": tune.randint(2, 20),
        "norm": True,
        "num_epochs": config.max_epochs,
        "seg_len": tune.choice([10, 50, 100, 200, 400]),
        "seq_len": 1600,
        "factor": tune.randint(1, 20),  # from ETTm1 script
        "patience": 10,
        "moving_avg": tune.choice([1, 3, 5]),
    }
    return search_space


def itransformer_searchspace():
    search_space = {
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3]),
        "dropout": tune.uniform(0.0001, 0.2),
        "horizon": 400,
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "norm": True,
        "num_epochs": config.max_epochs,
        "moving_avg": 1,
        "period_len": tune.choice([10, 50, 100, 200, 400]),
        "seq_len": 1600,
        "n_heads": tune.randint(1, 20),
        "patience": 10,
        "moving_avg": tune.choice([1, 3, 5]),
        # TODO: freq?
    }
    return search_space


def timexer_searchspace():
    search_space = {
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3]),
        "dropout": tune.uniform(0.0001, 0.2),
        "factor": tune.randint(1, 20),
        "features": "M",  # TimeXer specific: M for multivariate, S for univariate
        "horizon": 400,
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "n_heads": tune.randint(2, 20),
        "norm": True,  
        "num_epochs": config.max_epochs,
        "patch_len": tune.choice([10, 50, 100, 200, 400]),  # TimeXer specific patch length
        "patience": 10,
        "seq_len": 1600,
        "use_norm": True,  # TimeXer specific normalization parameter
        "moving_avg": tune.choice([1, 3, 5]),
    }
    return search_space


def dlinear_searchspace():
    search_space = {
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512, 1024, 2048]),
        "d_model": tune.choice([64, 128, 256, 512, 1024]),
        "dropout": tune.uniform(0.001, 0.2),
        "horizon": 400,
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "norm": True,
        "num_epochs": config.max_epochs,
        "moving_avg": tune.choice([1, 3, 5]),
        "seq_len": 1600,
        "patience": 10,
    }
    return search_space


def patchtst_searchspace():
    search_space = {
        "batch_size": 64,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3]),
        "dropout": tune.uniform(0.0001, 0.3),
        "horizon": 400,
        "loss": "MAE",
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": "type1",
        "norm": True,
        "num_epochs": config.max_epochs,
        "moving_avg": tune.choice([1, 3, 5]),
        "patch_len": tune.choice(
            [16, 50, 100, 160, 200]
        ),  # TODO better use values where 1600 lookback the is an even lookback
        "seq_len": 1600,
        "n_heads": tune.randint(1, 20),
        "patience": 10,
    }
    return search_space


def pdf_searchspace():
    search_space = {
        "batch_size": 32,
        "d_ff": tune.choice([64, 128, 256, 512]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "dropout": tune.uniform(0.001, 0.5),
        "e_layers": tune.choice([2, 3, 4, 5]),
        "fc_dropout": tune.uniform(0.001, 0.5),
        "kernel_list": [3, 5, 7, 11],
        "n_head": tune.randint(2, 20),
        "period_len": [100],
        "patch_len": [20],
        "stride": [4],
        "num_epochs": config.max_epochs,
        "lr": tune.loguniform(1e-5, 1e-2),
        "loss": "MAE",
        "moving_avg": tune.choice([1, 3, 5]),
        "horizon": 400,
        "seq_len": 1600,
        "lradj": "type1",
        "norm": True,
    }
    return search_space
