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

tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))
sys.path.append(tfb_path)
os.environ["PYTHONPATH"] = tfb_path
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ts_benchmark.baselines.pdf.models.PDF import Model as PDF_model
from ts_benchmark.data import data_source
from ts_benchmark.models.model_loader import get_models
from ts_benchmark.baselines.duet.utils.tools import adjust_learning_rate


class CustomDatasetWithOverrides(Dataset):
    def __init__(
        self,
        orig_data,
        override_data,
        nlookback,
        nlookahead,
        label_length,
        input_sampling=1,
        transform=None,
    ):
        self.orig_data = orig_data
        self.override_data = override_data
        self.transform = transform
        self.nlookback = nlookback
        self.nlookahead = nlookahead
        self.label_length = label_length  # Decoder lookback length
        self.input_sampling = input_sampling
        self.virtual_len = 2 * len(
            orig_data
        )  # Even split between orig_data and override_data
        self.overrides = self._find_trajectory_override_ranges(
            self.override_data["Override"].values,
            self.override_data["TargetYaw"].values,
        )
        self.cycle_start_idxs = self._find_cycle_starts(
            self.orig_data["TargetYaw"].values
        )

        self.overrides2 = self._adjust_override_start_idx(
            self.overrides, self.cycle_start_idxs
        )
        self.time_offset = 0

    def _adjust_override_start_idx(self, overrides, cycle_start_idxs):
        j = 0
        adjusted_overrides = []
        for override_start_idx, override_end_idx, cycle_start_before_idx in overrides:
            override_dist_to_start = override_start_idx - cycle_start_before_idx
            orig_data_start_idx = cycle_start_idxs[j] + override_dist_to_start
            adjusted_overrides.append(
                (override_start_idx, override_end_idx, orig_data_start_idx)
            )
            j += 1
            if j >= len(cycle_start_idxs):
                j = 0
        return adjusted_overrides

    def _find_cycle_starts(self, arr):
        cycle_start_val = np.min(arr)
        max_val = 0.0
        cycle_start_idx = -1
        start_idxs_and_max = []
        min_seconds = 5 * 100
        for i in range(0, len(arr)):
            if abs(arr[i] - cycle_start_val) < 1e-5:
                dist_to_last_cycle_start = i - cycle_start_idx
                if dist_to_last_cycle_start > min_seconds:
                    start_idxs_and_max.append((cycle_start_idx, max_val))
                    max_val = 0
                cycle_start_idx = i
            if arr[i] > max_val:
                max_val = arr[i]
        prev_max_per_cycle = 0

        first_cycles = []
        for start_idx, max_per_cycle in start_idxs_and_max:
            if max_per_cycle < prev_max_per_cycle:
                # New overall cycle (consisting of 4 cycles) starts wit the lowest "max"
                first_cycles.append(start_idx)
            prev_max_per_cycle = max_per_cycle
        return first_cycles

    def _find_trajectory_override_ranges(self, overrides, targetyaw):
        cycle_start_val = np.min(targetyaw)
        cycle_start_idx = 0
        last_cycle_start_idx = 0

        override_val = np.max(overrides)
        found_len = 0
        found_len_start = 0
        override_ranges = []
        override_start_idx = np.argmax(overrides > 0.9)

        for i in range(
            self.nlookback, len(self.override_data) - self.nlookback - self.nlookahead
        ):
            if abs(targetyaw[i] - cycle_start_val) < 1e-5:
                if cycle_start_idx != i - 1:
                    last_cycle_start_idx = cycle_start_idx
                cycle_start_idx = i

            if abs(overrides[i] - override_val) < 1e-5:
                if found_len == 0:
                    found_len_start = i
                found_len += 1
            elif found_len > 0:
                override_ranges.append((found_len_start, i, last_cycle_start_idx))
                found_len = 0
        return override_ranges

    def __len__(self):
        return (
            self.virtual_len
            - 2 * self.nlookback * self.input_sampling
            - 2 * self.nlookahead
        )

    def _find_next_override(self, offset):
        offset = int(offset / len(self.orig_data) * len(self.override_data))
        override_start_idx = self.overrides[0][0]  # Fallback
        orig_data_start_idx = self.overrides[0][2]

        for start_idx, end_idx, orig_data_start_idx in self.overrides:
            if offset < start_idx and start_idx > self.input_sampling * self.nlookback:
                override_start_idx = start_idx
                break
        slice_start_random_offset = offset % 100
        include_target_pitch_change_offset = 4
        slice_start = (
            override_start_idx
            - self.nlookback * self.input_sampling
            + include_target_pitch_change_offset
            + slice_start_random_offset
        )

        split_idx = slice_start + self.input_sampling * self.nlookback
        seq_lookback = self.override_data.iloc[
            slice_start : split_idx : self.input_sampling, :
        ].values
        seq_lookahead = self.override_data.iloc[
            split_idx - self.label_length : split_idx + self.nlookahead, :
        ].values

        if len(seq_lookback) < self.nlookback:
            print(
                "Error - insufficient len:",
                offset,
                len(self.orig_data) - self.nlookahead - self.nlookahead,
                override_start_idx,
            )
        return seq_lookback, seq_lookahead, orig_data_start_idx

    def __getitem__(self, idx):
        seq = None

        if (
            idx
            < len(self.orig_data)
            - self.nlookback * self.input_sampling
            - self.nlookahead
        ):
            split_idx = idx + self.nlookback * self.input_sampling
            seq_lookback = self.orig_data.iloc[
                idx : split_idx : self.input_sampling, :
            ].values
            seq_lookahead = self.orig_data.iloc[
                split_idx - self.label_length : split_idx + self.nlookahead, :
            ].values

            if len(seq_lookback) < self.nlookback:
                print(
                    "Error, sequence too short!",
                    len(seq_lookback),
                    self.nlookback,
                    self.nlookahead,
                )
        else:
            idx -= (
                len(self.orig_data)
                - self.nlookback * self.input_sampling
                - self.nlookahead
            )
            seq_lookback, seq_lookahead, idx = self._find_next_override(idx)

        if self.transform:
            seq_lookback = self.transform(
                seq_lookback
            )  # TODO: This is not great, preformance-wise
            seq_lookahead = self.transform(seq_lookahead)

        lookback_actual = self.nlookback * self.input_sampling
        seq_lookback_mark = (
            torch.arange(0, lookback_actual, self.input_sampling)
            + idx
            + self.time_offset
        ).unsqueeze(-1)
        seq_lookahead_mark = (
            torch.arange(
                lookback_actual - self.label_length, lookback_actual + self.nlookahead
            )
            + idx
            + self.time_offset
        ).unsqueeze(-1)
        seq_lookback = torch.Tensor(seq_lookback)
        seq_lookahead = torch.Tensor(seq_lookahead)

        return seq_lookback, seq_lookahead, seq_lookback_mark, seq_lookahead_mark


# From TFB
def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.lr}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def forecast_fit(model, train_dataset, validate_dataset, **kwargs) -> "ModelBase":

    if model.model is None:
        raise ValueError("Model not trained. Call the fit() function first.")
    if kwargs.get("use_checkpoint", True):
        model.model.load_state_dict(model.early_stopping.check_point)

    if not kwargs.get("already_transformed", False) and model.config.norm:
        train_dataset.transform = model.scaler.transform
        validate_dataset.transform = model.scaler.transform

    config = model.config

    train_data_loader = DataLoader(
        train_dataset,
        num_workers=1,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        validate_dataset,
        num_workers=1,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
    )

    if config.loss == "MSE":
        criterion = nn.MSELoss()
    elif config.loss == "MAE":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=0.5)

    optimizer = optim.Adam(model.model.parameters(), lr=config.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.early_stopping = common.EarlyStopping(patience=config.patience)
    model.model.to(device)
    total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params}")

    for epoch in range(config.num_epochs):
        if kwargs.get("random_time_offset_per_epoch", False):
            time_offset = np.random.randint(2**14, size=1)[0]
            train_data_loader.dataset.time_offset = time_offset
            valid_data_loader.dataset.time_offset = time_offset

        model.model.train()
        for i, (input, target, input_mark, target_mark) in enumerate(train_data_loader):
            optimizer.zero_grad()
            input, target, input_mark, target_mark = (
                input.to(device),
                target.to(device),
                input_mark.to(device),
                target_mark.to(device),
            )
            if model.config.decoder_input_required:
                dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(device)
                )
                output = model.model(input, input_mark, dec_input, target_mark)
            elif model.config.has_loss_importance:
                output, loss_importance = model.model(input)
            else:
                output = model.model(input)

            target = target[:, -config.horizon :, :]
            output = output[:, -config.horizon :, :]
            loss = criterion(output, target)

            if model.config.has_loss_importance:
                loss = loss + loss_importance

            loss.backward()
            optimizer.step()

        # DeepForecastingModelBase.validate signature: validate(valid_data_loader, series_dim, criterion)
        # Our custom training loop previously omitted series_dim, causing a TypeError.
        series_dim = getattr(model.config, 'c_out', None)
        if series_dim is None:
            # Fallback: infer from batch (target shape: [B, T, D])
            first_batch = next(iter(valid_data_loader))
            series_dim = first_batch[1].shape[-1]
        try:
            valid_loss = model.validate(valid_data_loader, series_dim, criterion)
        except TypeError:
            # In case some models use old signature validate(valid_data_loader, criterion)
            valid_loss = model.validate(valid_data_loader, criterion)  # type: ignore

        callback_fcn = kwargs.get("training_progress_callback", None)
        if callback_fcn:
            callback_fcn(valid_loss, model.early_stopping.val_loss_min, model)

        if model.early_stopping(valid_loss, model.model):
            break

        adjust_learning_rate(optimizer, epoch + 1, config)


def train_model(model_config, evaluation_config, **kwargs):
    if "seed" in evaluation_config:
        common.set_fixed_seed(evaluation_config["seed"])

    model_factory = get_models(model_config)[0]
    model = model_factory()
    for key, val in model_config["models"][0].items():
        model.config.__setattr__(key, val)
    for key, val in model.config.__dict__.items():
        print(key, val)

    data = common.load_csv(kwargs["data_path"])
    data_w_overrides = common.load_csv(kwargs["data_path_overrides"])

    setattr(model.config, "task_name", "short_term_forecast")
    model.multi_forecasting_hyper_param_tune(data)

    # TODO: each model needs to be constructable the same ...
    if "duet" in model_config["models"][0]["model_name"]:
        model.model = DUETModel(model.config)
    elif "pdf" in model_config["models"][0]["model_name"]:
        model.model = PDF_model(model.config)
    else:
        model.model = model.model_class(model.config)

    lookback = model_config["models"][0]["model_hyper_params"]["seq_len"]
    horizon = evaluation_config["strategy_args"]["horizon"]
    lookahead = horizon

    train_ratio_in_tv = evaluation_config["strategy_args"]["train_ratio_in_tv"]
    tv_ratio = evaluation_config["strategy_args"][
        "tv_ratio"
    ]  # train+validate ratio, remaining is for testing

    train_data, valid_data, _ = common.split_data(data, tv_ratio, train_ratio_in_tv)
    train_data_w_overrides, valid_data_w_overrides, _ = common.split_data(
        data_w_overrides, tv_ratio, train_ratio_in_tv
    )

    model.scaler.partial_fit(train_data.values)
    model.scaler.partial_fit(train_data_w_overrides.values)

    train_data = common.scaled_dataframe_copy(train_data, model.scaler)
    valid_data = common.scaled_dataframe_copy(valid_data, model.scaler)
    train_data_w_overrides = common.scaled_dataframe_copy(
        train_data_w_overrides, model.scaler
    )
    valid_data_w_overrides = common.scaled_dataframe_copy(
        valid_data_w_overrides, model.scaler
    )

    gc.collect()

    train_dataset = CustomDatasetWithOverrides(
        train_data,
        train_data_w_overrides,
        lookback,
        lookahead,
        model.config.label_len,
        model.config.input_sampling,
    )
    validate_dataset = CustomDatasetWithOverrides(
        valid_data,
        valid_data_w_overrides,
        lookback,
        lookahead,
        model.config.label_len,
        model.config.input_sampling,
    )

    forecast_fit(
        model,
        train_dataset,
        validate_dataset,
        use_checkpoint=False,
        random_time_offset_per_epoch=True,
        already_transformed=True,
        training_progress_callback=kwargs.get("training_progress_callback", None),
    )
