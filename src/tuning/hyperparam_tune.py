# -*- coding: utf-8 -*-
import argparse
import copy
import gc
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import warnings
from typing import Dict, NoReturn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import Checkpoint, CheckpointConfig

import common
import config
import hyperparam_search_spaces
import training

tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))
sys.path.append(tfb_path)
os.environ["PYTHONPATH"] = tfb_path
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ts_benchmark.data import data_source
from ts_benchmark.models.model_loader import get_models

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep config")

    parser.add_argument(
        "--model-setup",
        type=str,
        required=True,
        help="Model setup as defined in hyperparam_search_spaces.py",
    )
    parser.add_argument("--sweep-name", type=str, default=None, help="Sweep name")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Number of hyperparameter configurations to try",
    )
    parser.add_argument(
        "--cpu-per-trial",
        type=float,
        default=1.0,
        help="Number of CPU resources per trial",
    )
    parser.add_argument(
        "--gpu-per-trial",
        type=float,
        default=4 / 12,
        help="Number of GPU resources to allocate per trial (1/3 -> 3 trials will share 1 GPU)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/pick_n_place_procedure_dataset.csv",
        help="Path to the main dataset CSV file.",
    )
    parser.add_argument(
        "--data-path-overrides",
        type=str,
        default="datasets/pick_n_place_procedure_w_overrides.csv",
        help="Path to the dataset with overrides CSV file.",
    )
    return parser.parse_args()


def callback(val_loss, val_loss_min, model):
    print("-> Callback. Validation loss:  %f" % (val_loss))
    if val_loss < val_loss_min:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_file = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            common.save_model(model, model_file)
            print(model_file)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"val_loss": val_loss}, checkpoint=checkpoint)
    else:
        train.report({"val_loss": val_loss}, checkpoint=None)


def start_training(selected_hyperparams, model_config, evaluation_config, args):

    model_config["models"][0]["model_hyper_params"] = selected_hyperparams
    training.train_model(
        model_config,
        evaluation_config,
        training_progress_callback=callback,
        data_path=args.data_path,
        data_path_overrides=args.data_path_overrides,
    )


if __name__ == "__main__":
    args = parse_args()

    eval_config, model_config, search_space = hyperparam_search_spaces.assemble_setup(
        args.model_setup
    )
    model_config["models"][0]["input_sampling"] = 1

    scheduler = ASHAScheduler(
        max_t=config.max_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    previous_good_params = None

    search_alg = HyperOptSearch(
        metric="val_loss", mode="min", points_to_evaluate=previous_good_params
    )

    ray.init()

    tune.register_trainable(
        args.sweep_name,
        lambda selected_hyperparams: start_training(
            selected_hyperparams, model_config, eval_config, args
        ),
    )

    analysis = tune.run(
        args.sweep_name,
        config=search_space,
        scheduler=scheduler,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
        search_alg=search_alg,
        metric="val_loss",
        mode="min",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )

    print(analysis.results_df)
    print(analysis.best_result_df)
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    print(
        "Best checkpoint: ",
        analysis.get_best_checkpoint(best_trial, metric="val_loss", mode="min"),
    )
    print(
        "Best hyperparameters: ",
        analysis.get_best_config(metric="val_loss", mode="min", scope="all"),
    )
