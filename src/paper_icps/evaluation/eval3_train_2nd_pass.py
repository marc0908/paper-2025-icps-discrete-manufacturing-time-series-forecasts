import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core import common, config, training, dataset


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
            model, lookback,
            use_best_checkpoint=False,
            move_model=False,
        )
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


# eval3_train_2nd_pass.py  (add this near the bottom, above if __name__ == "__main__":)

def run_second_training(
    model_path: str,
    data_path: str,
    data_path_overrides: str,
    reduction_factor: int = 3,
    num_epochs: int = 100,
    learning_rate: float | None = None,
    patience: int = 10,
    seed: int | None = None,
    save_suffix: str = "",
) -> str:
    """
    Loads model from model_path, performs the 2nd-pass training, saves a new model,
    and returns the new model path.
    """

    data = common.load_csv(data_path)
    data_w_overrides = common.load_csv(data_path_overrides)
    model = common.restore_model(model_path)

    if hasattr(model, "model") and model.model is not None:
        model.model.eval()
        model.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    lookback_len = model.config.seq_len
    generated_len_per_run = model.config.horizon
    lookahead_len = generated_len_per_run

    cfg = config.default_eval_config()
    if seed is None:
        seed = cfg["strategy_args"]["seed"]
    common.set_fixed_seed(seed)

    train_ratio_in_tv = cfg["strategy_args"]["train_ratio_in_tv"]
    tv_ratio = cfg["strategy_args"]["tv_ratio"]

    train_data, valid_data, _ = common.split_data(data, tv_ratio, train_ratio_in_tv)
    train_data_w_overrides, valid_data_w_overrides, _ = common.split_data(
        data_w_overrides, tv_ratio, train_ratio_in_tv
    )

    n_generated_train = int(len(train_data) // reduction_factor)
    n_generated_valid = int(len(valid_data) // reduction_factor)

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
            size=n_generated_valid,
        )
    )

    generated_results = []
    generated_results_valid = []
    n_recursion = 1

    print(f"[eval3] Generating {n_generated_train} trajectories for training ...")
    for s in trajectory_starts:
        in_data = train_data[s:].values
        generated = generate_forecast(model, in_data, n_recursion)
        generated_results.append(model.scaler.transform(generated))

    print(f"[eval3] Generating {n_generated_valid} trajectories for validation ...")
    for s in trajectory_starts_valid:
        in_data = valid_data[s:].values
        generated = generate_forecast(model, in_data, n_recursion)
        generated_results_valid.append(model.scaler.transform(generated))

    # scale original data
    train_data = common.scaled_dataframe_copy(train_data, model.scaler)
    valid_data = common.scaled_dataframe_copy(valid_data, model.scaler)
    train_data_w_overrides = common.scaled_dataframe_copy(train_data_w_overrides, model.scaler)
    valid_data_w_overrides = common.scaled_dataframe_copy(valid_data_w_overrides, model.scaler)

    # datasets
    train_dataset = MixerDataset(
        train_data, generated_results, trajectory_starts,
        lookback_len, generated_len_per_run,
        getattr(model.config, "label_len", 0),
    )
    validate_dataset = MixerDataset(
        valid_data, generated_results_valid, trajectory_starts_valid,
        lookback_len, generated_len_per_run,
        getattr(model.config, "label_len", 0),
    )

    orig_train_dataset = dataset.CustomDatasetWithOverrides(
        train_data, train_data_w_overrides,
        lookback_len, lookahead_len,
        getattr(model.config, "label_len", 0),
    )
    orig_validate_dataset = dataset.CustomDatasetWithOverrides(
        valid_data, valid_data_w_overrides,
        lookback_len, lookahead_len,
        getattr(model.config, "label_len", 0),
    )

    mixed_train_dataset = TorchMultiDataset(train_dataset, orig_train_dataset)
    mixed_validate_dataset = TorchMultiDataset(validate_dataset, orig_validate_dataset)

    # training hyperparams
    model.config.num_epochs = num_epochs
    model.config.patience = patience
    if learning_rate is not None:
        model.config.lr = learning_rate

    # train
    training.forecast_fit(
        model,
        mixed_train_dataset,
        mixed_validate_dataset,
        use_checkpoint=True,
        already_transformed=True,
        training_progress_callback=callback,
    )

    # save
    model_path_wo_ext, _ = os.path.splitext(model_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{save_suffix}" if save_suffix else ""
    out_path = model_path_wo_ext + f"{suffix}_with_2nd_training_{timestamp}.pt"
    common.save_model(model, out_path)
    print("[eval3] Model saved under:", out_path)

    return out_path