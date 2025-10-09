import copy
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import torch

tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))
sys.path.append(tfb_path)
from paper_icps.TFB.ts_benchmark.baselines.duet.models.duet_model import DUETModel
from paper_icps.TFB.ts_benchmark.data import data_source
from paper_icps.TFB.ts_benchmark.models.model_loader import get_models


default_seed = 2025


def set_fixed_seed(seed=default_seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass


def columns_to_use():
    return [
        "Voltage0",
        "Voltage1",
        "PitchDot",
        "YawDot",
        "Pitch",
        "Yaw",
        "TargetPitch",
        "TargetYaw",
        "Override",
    ]


def load_csv(file_path):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    df = pd.read_csv(file_path, delimiter=",", header=0)

    df = df[columns_to_use()]
    # TODO: hardcoded 100 Hz
    df["date"] = pd.date_range(
        start="2024-02-01", periods=df.shape[0], freq=str(10) + "ms", unit="ms"
    )
    df.set_index("date", inplace=True)

    validate_dataframe_schema(df, dataset_name=os.path.basename(file_path))
    return df


def validate_dataframe_schema(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """Validate that required columns exist and are numeric.

    Raises a ValueError with a clear message if validation fails.
    """
    required_cols = columns_to_use()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: Missing required columns: {missing}. Present: {list(df.columns)}"
        )

    non_numeric = []
    for col in required_cols:
        if not is_numeric_dtype(df[col]):
            # attempt to coerce to numeric without modifying original df
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isna().any():
                non_numeric.append(col)
            else:
                # safe to coerce in-place if no NaNs introduced
                df[col] = coerced
    if non_numeric:
        raise ValueError(
            f"{dataset_name}: Non-numeric values detected in columns: {non_numeric}. "
            f"Please ensure these columns are numeric."
        )


def resolve_data_paths(data_path=None, data_path_overrides=None):
    """Resolve dataset paths with environment overrides and safe defaults."""
    default_main = "../dataset/pick_n_place_procedure_dataset.csv"
    default_overrides = "../dataset/pick_n_place_procedure_w_overrides.csv"

    env_main = os.environ.get("DATA_PATH")
    env_over = os.environ.get("DATA_PATH_OVERRIDES")

    final_main = data_path or env_main or default_main
    final_over = data_path_overrides or env_over or default_overrides
    return final_main, final_over


def split_data(data, tv_ratio, train_ratio_in_tv):
    train_length = int(tv_ratio * len(data))
    test_length = len(data) - train_length
    train_valid_data, test_data = (
        data.iloc[:train_length, :],
        data.iloc[train_length:, :],
    )

    train_only_len = int(train_length * train_ratio_in_tv)
    train_data = data.iloc[:train_only_len, :]
    valid_data = data.iloc[train_only_len:train_length, :]
    test_data = data.iloc[train_length:, :]
    return train_data, valid_data, test_data


def scaled_dataframe_copy(dataframe, scaler):
    return pd.DataFrame(
        scaler.transform(dataframe.values),
        columns=dataframe.columns,
        index=dataframe.index,
    )


def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def restore_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def sum_model_params(model):
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


def forecast_custom(model, history: np.ndarray, use_best_checkpoint=True) -> np.ndarray:

    if use_best_checkpoint and hasattr(model, 'early_stopping') and model.early_stopping.check_point is not None:
        model.model.load_state_dict(model.early_stopping.check_point)

    if model.config.norm:
        history = model.scaler.transform(history)

    if model.model is None:
        raise ValueError("Model not trained. Call the fit() function first.")

    config = model.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.model.eval()
    model.model.to(device)

    model_forecast_len = config.horizon
    model_history_len = config.seq_len
    history = torch.tensor(
        [history[-model_history_len:]], dtype=torch.float
    )  # use the last values according to model_history_len
    history = history.to(device)
    result = None

    if not hasattr(model.config, "decoder_input_required"):
        # TODO: Retrain - This is a legacy version of DUET
        setattr(model.config, "decoder_input_required", False)
        setattr(model.config, "has_loss_importance", True)

    if model.config.decoder_input_required:
        target = history
        dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
        dec_input = (
            torch.cat([target[:, -config.label_len :, :], dec_input], dim=1)
            .float()
            .to(device)
        )
        history_mark = (
            torch.arange(0, model_history_len).unsqueeze(0).unsqueeze(-1)
        )  # self.data_stamp[s_begin:s_end] # Not needed for now
        dec_input_mark = (
            torch.arange(
                model_history_len - config.label_len,
                model_history_len + model_forecast_len,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # self.data_stamp[r_begin:r_end] # Not needed for now
        with torch.no_grad():
            dec_input = dec_input.to(device)
            history_mark = history_mark.to(device)
            dec_input_mark = dec_input_mark.to(device)
            result = model.model(history, history_mark, dec_input, dec_input_mark)

    elif model.config.has_loss_importance:
        with torch.no_grad():
            result, _ = model.model(history)
    else:
        with torch.no_grad():
            result = model.model(history)

    result = result.cpu().numpy()[0]

    if model.config.norm:
        result = model.scaler.inverse_transform(result)
    return result


# From TFB
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.check_point = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
