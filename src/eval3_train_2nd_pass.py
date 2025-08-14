import argparse
import datetime
import io
import os
import pickle
import pdb
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.utils import resample
from torch.utils.data import Dataset

import common
import config
import training

tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))
sys.path.append(tfb_path)
from ts_benchmark.data import data_source
from ts_benchmark.models.model_loader import get_model_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform a second training with original data and own model forecasts."
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
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="Data reduction factor for the model training (default: 3).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override (optional).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10).",
    )

    return parser.parse_args()


nepochs = 0


def callback(val_loss, val_loss_min, model):
    global nepochs
    print("-> Callback: Epoch: %i, validation loss: %f " % (nepochs, val_loss))
    nepochs += 1


def generate_forecast(model, data, n_recursion):
    lookback_len = model.config.seq_len
    lookback_start_idx = 0

    generated_len_per_run = model.config.horizon
    initial_lookback = data[:lookback_len]
    lookback = initial_lookback.copy()
    time_per_generation = []
    generated_results = []

    for i in range(n_recursion):
        start_time = time.time()
        generated = common.forecast_custom(
            model, lookback
        )  # TODO: for downsampling, pass dec input with full sampling rate
        time_per_generation.append(time.time() - start_time)
        lookback = np.vstack((lookback[len(generated) :, :], generated))
        generated_results.append(generated)

    generated = np.vstack(generated_results)
    return generated


class MixerDataset(Dataset):
    def __init__(
        self, data, generated, generated_idx, nlookback, nlookahead, label_length
    ):
        self.data = data
        self.generated = generated
        self.generated_idx = generated_idx
        self.nlookback = nlookback
        self.nlookahead = nlookahead
        self.label_length = label_length

    def __len__(self):
        return len(self.generated)

    def __getitem__(self, idx):
        pos = self.generated_idx[idx] + self.nlookahead

        seq = self.data.iloc[pos : pos + self.nlookahead + self.nlookback, :].values
        seq[self.nlookback - self.nlookahead : self.nlookback] = self.generated[idx]
        seq_lookback = seq[: self.nlookback, :]
        seq_lookahead = seq[-self.nlookahead - self.label_length :, :]
        seq_lookback_mark = torch.arange(0, self.nlookback).unsqueeze(-1)
        seq_lookahead_mark = torch.arange(
            self.nlookback - self.label_length, self.nlookback + self.nlookahead
        ).unsqueeze(-1)
        seq_lookback = torch.Tensor(seq_lookback)
        seq_lookahead = torch.Tensor(seq_lookahead)

        return seq_lookback, seq_lookahead, seq_lookback_mark, seq_lookahead_mark


class TorchMultiDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.total_len = self.len1 + self.len2
        sample = dataset1[0]
        self.expected_shapes = tuple(t.shape for t in sample)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.len1:
            item = self.dataset1[idx]
            source = "dataset1"
        else:
            item = self.dataset2[idx - self.len1]
            source = "dataset2"

        for i, (tensor, expected_shape) in enumerate(zip(item, self.expected_shapes)):
            if tensor.shape != expected_shape:
                print(
                    f"[Shape Mismatch] {source} idx={idx}, tensor[{i}] shape={tensor.shape}, expected={expected_shape}"
                )

        return item


if __name__ == "__main__":
    args = parse_args()

    data = common.load_csv(args.data_path)
    data_w_overrides = common.load_csv(args.data_path_overrides)
    model = common.restore_model(args.model_path)

    lookback_len = model.config.seq_len
    generated_len_per_run = model.config.horizon
    lookahead_len = generated_len_per_run

    cfg = config.default_eval_config()
    seed = cfg["strategy_args"]["seed"]
    common.set_fixed_seed(seed)

    train_ratio_in_tv = cfg["strategy_args"]["train_ratio_in_tv"]
    tv_ratio = cfg["strategy_args"]["tv_ratio"]

    train_data, valid_data, test_data = common.split_data(
        data, tv_ratio, train_ratio_in_tv
    )
    train_data_w_overrides, valid_data_w_overrides, _ = common.split_data(
        data_w_overrides, tv_ratio, train_ratio_in_tv
    )

    n_generated_train = len(train_data) // args.reduction_factor
    n_generated_valid = len(valid_data) // args.reduction_factor

    rng = np.random.default_rng(seed=seed)

    trajectory_starts = list(
        rng.integers(
            low=0,
            high=len(train_data) - lookback_len - generated_len_per_run,
            size=n_generated_train,
        )
    )
    trajectory_starts_valid = list(
        rng.integers(
            low=0,
            high=len(valid_data) - lookback_len - generated_len_per_run,
            size=int(n_generated_valid),
        )
    )

    generated_results = []
    generated_results_valid = []
    n_recursion = 1

    print(f"Generating {n_generated_train} trajectories for training ...")
    for trajectory_start_idx in trajectory_starts:
        in_data = train_data[trajectory_start_idx:].values
        generated = generate_forecast(model, in_data, n_recursion)
        generated_results.append(model.scaler.transform(generated))

    print(f"Generating {n_generated_valid} trajectories for validation ...")
    for trajectory_start_idx in trajectory_starts_valid:
        in_data = valid_data[trajectory_start_idx:].values
        generated = generate_forecast(model, in_data, n_recursion)
        generated_results_valid.append(model.scaler.transform(generated))

    train_data = common.scaled_dataframe_copy(train_data, model.scaler)
    valid_data = common.scaled_dataframe_copy(valid_data, model.scaler)
    train_data_w_overrides = common.scaled_dataframe_copy(
        train_data_w_overrides, model.scaler
    )
    valid_data_w_overrides = common.scaled_dataframe_copy(
        valid_data_w_overrides, model.scaler
    )

    train_dataset = MixerDataset(
        train_data,
        generated_results,
        trajectory_starts,
        lookback_len,
        generated_len_per_run,
        model.config.label_len,
    )
    validate_dataset = MixerDataset(
        valid_data,
        generated_results_valid,
        trajectory_starts_valid,
        lookback_len,
        generated_len_per_run,
        model.config.label_len,
    )

    orig_train_dataset = training.CustomDatasetWithOverrides(
        train_data,
        train_data_w_overrides,
        lookback_len,
        lookahead_len,
        model.config.label_len,
    )
    orig_validate_dataset = training.CustomDatasetWithOverrides(
        valid_data,
        valid_data_w_overrides,
        lookback_len,
        lookahead_len,
        model.config.label_len,
    )

    mixed_train_dataset = TorchMultiDataset(train_dataset, orig_train_dataset)
    mixed_validate_dataset = TorchMultiDataset(validate_dataset, orig_validate_dataset)

    model.config.num_epochs = args.num_epochs
    model.config.patience = args.patience
    if args.learning_rate:
        model.config.lr = args.learning_rate

    training.forecast_fit(
        model,
        mixed_train_dataset,
        mixed_validate_dataset,
        use_checkpoint=True,
        already_transformed=True,
        training_progress_callback=callback,
    )

    model_path_wo_ext, _ = os.path.splitext(args.model_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_save_path = model_path_wo_ext + f"_with_2nd_training_{timestamp}.pt"
    common.save_model(model, model_save_path)
    print("Model saved under: ", model_save_path)
