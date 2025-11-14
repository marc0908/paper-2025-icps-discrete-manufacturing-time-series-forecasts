import copy
import os
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple, IO, Any, Union

import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from pandas.api.types import is_numeric_dtype
import torch

default_seed = 2025


def set_fixed_seed(seed=default_seed):
    """
    Set global random seeds for reproducibility across Python, NumPy, and PyTorch.
    Enables deterministic CUDA behavior when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            import torch.backends.cudnn as cudnn

            # Force cuDNN to use deterministic kernels (slightly slower, fully reproducible)
            cudnn.deterministic = True

            # Disable cuDNN auto-tuning to avoid nondeterministic algorithm selection
            cudnn.benchmark = False
        except ImportError:
            # cuDNN backend not available; skip deterministic settings
            pass


# Columns used as canonical features for all models.
COLUMNS_TO_USE = [
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


def load_csv(
        file_path: Union[str, Path],
        freq_hz: int = 100,
        start_time: str = "2024-02-01"
) -> pd.DataFrame:
    """
    Load a CSV relative to this module, select the canonical feature columns,
    attach a synthetic datetime index based on a fixed sampling rate, validate
    the schema, and return the resulting DataFrame.

    Parameters
    ----------
    file_path : str
        Relative path to the CSV file inside the package directory.

    freq_hz : int, optional (default=100)
        Sampling frequency of the dataset in Hertz. Used to generate the
        datetime index (e.g., 100 Hz → 10 ms steps).

    start_time : str, optional
        Start timestamp used for the synthetic datetime index.
    """
    # Resolve path relative to this file to make loading robust to CWD changes
    csv_path = Path(__file__).resolve().parent / file_path

    df = pd.read_csv(csv_path, delimiter=",", header=0)

    # Restrict to the expected feature columns and preserve column order
    df = df.loc[:, COLUMNS_TO_USE].copy()

    # Convert Hz → milliseconds per sample (e.g., 100 Hz → 10 ms)
    step_ms = int(1000 / freq_hz)

    # Generate a synthetic datetime index at the specified sampling rate
    df["date"] = pd.date_range(
        start=start_time,
        periods=len(df),
        freq=f"{step_ms}ms",
        unit="ms",
    )
    df.set_index("date", inplace=True)

    # Ensure that the loaded data matches the expected schema (columns, dtypes, index)
    validate_dataframe_schema(df, dataset_name=csv_path.name)

    return df


def validate_dataframe_schema(
    df: pd.DataFrame,
    dataset_name: str = "dataset"
) -> None:
    """
    Validate that the DataFrame contains the required columns and that those
    columns can be interpreted as numeric. Raises a ValueError with a clear,
    actionable message if validation fails.
    """

    # Check for missing required columns
    missing = [col for col in COLUMNS_TO_USE if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: Missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )

    # Identify columns that cannot be coerced to numeric values
    non_numeric = []
    for col in COLUMNS_TO_USE:
        if not is_numeric_dtype(df[col]):
            # Try to convert without modifying the original column
            coerced = pd.to_numeric(df[col], errors="coerce")

            if coerced.isna().any():
                # NaNs indicate invalid numeric conversion → fail this column
                non_numeric.append(col)
            else:
                # Safe to convert in-place (no data loss)
                df[col] = coerced

    if non_numeric:
        raise ValueError(
            f"{dataset_name}: Non-numeric values detected in columns: {non_numeric}. "
            "Ensure all feature columns contain numeric, convertible data."
        )


def resolve_data_paths(
    data_path: Optional[str] = None,
    data_path_overrides: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Resolve dataset file paths based on (in order of priority):
    1. Explicit function arguments
    2. Environment variables
    3. Built-in default locations

    Returns fully resolved Paths to the main dataset and the overrides dataset.
    """

    # Built-in defaults (relative to project root)
    default_main = Path("../dataset/pick_n_place_procedure_dataset.csv")
    default_overrides = Path("../dataset/pick_n_place_procedure_w_overrides.csv")

    # Optional environment overrides (e.g., for SLURM, Docker, cloud runs)
    env_main = os.environ.get("DATA_PATH")
    env_over = os.environ.get("DATA_PATH_OVERRIDES")

    # Resolution priority:
    # explicit argument > env var > default
    final_main = Path(data_path or env_main or default_main)
    final_over = Path(data_path_overrides or env_over or default_overrides)

    return final_main, final_over


def split_data(
        data: pd.DataFrame,
        tv_ratio: float,
        train_ratio_in_tv: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test segments using two ratios:

    Parameters
    ----------
    data : pd.DataFrame
        Full time-ordered dataset.
    tv_ratio : float
        Fraction of the dataset allocated to train+validation.
        Example: 0.8 → 80% train+val, 20% test.
    train_ratio_in_tv : float
        Fraction of the train+val portion allocated to *train*.
        Example: 0.75 → 75% train, 25% validation (within the 80%).

    Returns
    -------
    train_data, valid_data, test_data : pd.DataFrame
    """

    # Sanity check to avoid weird splits
    if not (0.0 < tv_ratio <= 1.0):
        raise ValueError(f"tv_ratio must be in (0, 1], got {tv_ratio}")
    if not (0.0 < train_ratio_in_tv <= 1.0):
        raise ValueError(f"train_ratio_in_tv must be in (0, 1], got {train_ratio_in_tv}")

    total_len = len(data)

    # Size of the train+validation region
    tv_len = int(tv_ratio * total_len)

    # Size of the training segment within the tv region
    train_len = int(tv_len * train_ratio_in_tv)

    # Segment the dataset using time-order (important for forecasting tasks)
    train_data = data.iloc[:train_len]
    valid_data = data.iloc[train_len:tv_len]
    test_data = data.iloc[tv_len:]

    return train_data, valid_data, test_data


def scaled_dataframe_copy(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Return a scaled copy of a DataFrame using a fitted scaler.
    Preserves column names and index.
    """
    # Using df.values ensures raw NumPy array input to the scaler
    scaled = scaler.transform(df.values)

    return pd.DataFrame(
        scaled,
        columns=df.columns,
        index=df.index,
    )


# noinspection PyTypeChecker
def save_model(model, file_path: str) -> None:
    """
    Persist a Python model (e.g., PyTorch, sklearn, or custom class)
    using pickle with the highest protocol.
    """
    with open(file_path, "wb") as f: # type: IO[bytes]
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def restore_model(file_path: str) -> Any:
    """
    Load a model saved with `save_model`.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def sum_model_params(model) -> int:
    """
    Count the number of trainable parameters of a PyTorch model.
    Assumes the model is accessible under `model.model`.
    """
    torch_model = getattr(model, "model", model)
    return sum(p.numel() for p in torch_model.parameters() if p.requires_grad)


def forecast_custom(model, history: np.ndarray, use_best_checkpoint : bool = True) -> np.ndarray:
    """
    Run a single-step (or multi-step) forecast using a trained model and a
    given history window. Handles normalization, best-checkpoint restoration,
    and model-specific decoder inputs.
    """

    # Optionally restore the best checkpoint saved by early stopping
    if (
            use_best_checkpoint
            and hasattr(model, "early_stopping")
            and model.early_stopping.check_point is not None
    ):
        model.model.load_state_dict(model.early_stopping.check_point)

    # Apply input normalization if the model was trained on normalized data
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

    # Use only the last `model_history_len` timesteps as model input
    history = torch.tensor(
        [history[-model_history_len:]], dtype=torch.float
    ).to(device)

    result = None

    # Backward-compatibility for legacy DUET configs missing these flags
    if not hasattr(model.config, "decoder_input_required"):
        # TODO: Retrain - This is a legacy version of DUET
        setattr(model.config, "decoder_input_required", False)
        setattr(model.config, "has_loss_importance", True)

    if model.config.decoder_input_required:
        # Encoder input
        target = history

        # Decoder input: last `label_len` encoder steps + zeroed forecast horizon
        dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
        dec_input = (
            torch.cat([target[:, -config.label_len :, :], dec_input], dim=1)
            .float()
            .to(device)
        )
        # Simple positional/time indices as "time marks"
        history_mark = torch.arange(0, model_history_len).unsqueeze(0).unsqueeze(-1)
        dec_input_mark = (
            torch.arange(
                model_history_len - config.label_len,
                model_history_len + model_forecast_len,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.no_grad():
            history_mark = history_mark.to(device)
            dec_input_mark = dec_input_mark.to(device)

            # DUET-style forward: (enc, enc_mark, dec, dec_mark)
            result = model.model(history, history_mark, dec_input, dec_input_mark)

    elif model.config.has_loss_importance:
        # Some models return (output, extra_loss_info)
        with torch.no_grad():
            result, _ = model.model(history)
    else:
        # Standard forward: output = model(history)
        with torch.no_grad():
            result = model.model(history)

    # Remove batch dimension and move back to NumPy
    result = result.cpu().numpy()[0]

    # De-normalize forecast back to original scale if normalization was applied
    if model.config.norm:
        result = model.scaler.inverse_transform(result)
    return result


# From TFB
class EarlyStopping:
    """
    Simple early stopping mechanism to halt training when the validation loss
    stops improving. Stores the best model checkpoint for later restoration.

    Parameters
    ----------
    patience : int
        Number of validation checks without improvement before stopping.
    delta : float
        Minimum improvement required to reset the counter.
    """

    def __init__(self, patience: int = 7, delta: float = 0.0):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

        # Stores the model state_dict of the best validation performance
        self.check_point = None

    def __call__(self, val_loss, model) -> bool:
        """
        Evaluate the current validation loss and determine whether to stop.

        Returns
        -------
        bool
            True if early stopping should trigger, False otherwise.
        """

        # Using negative validation loss because lower loss = higher score
        score = -val_loss
        if self.best_score is None:
            # First validation check → always accept
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # No sufficient improvement
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation improved → save new checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """
            Save a deep copy of the model state when validation improves.
        """
        print(
            f"Validation loss decreased "
            f"({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model..."
        )

        # Deep copy ensures no mutation even if the model continues training
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
