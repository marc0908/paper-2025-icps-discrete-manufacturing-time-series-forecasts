import gc
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol, cast, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from ..core import common
from ..TFB.ts_benchmark.models.model_loader import get_models

class ForecastingModel(Protocol):
    """
    Structural type for the forecasting models used in this project.

    Any object that:
      - wraps a torch.nn.Module in `.model`
      - has a `.config` with hyperparameters
      - has a `.scaler` for normalization
      - exposes `.early_stopping`
      - implements `validate(...)`
      - implements `multi_forecasting_hyper_param_tune(...)`
      - optionally provides `_init_model()` and `model_class`
    will satisfy this protocol, regardless of its actual base class.
    """

    model: nn.Module
    config: Any
    scaler: Any
    early_stopping: Any

    def validate(self, *args: Any, **kwargs: Any) -> float:
        ...

    def multi_forecasting_hyper_param_tune(self, data: Any) -> None:
        ...

    # These are used in train_model to construct the underlying nn.Module
    def _init_model(self) -> nn.Module:
        ...

    @property
    def model_class(self) -> Any:
        ...

class CustomDatasetWithOverrides(Dataset):
    """
    Dataset that mixes 'normal' trajectories and 'override' trajectories.

    It produces encoder/decoder input windows and corresponding "time marks"
    for models like DUET that use positional/time encodings as a separate input.

    For the first part of indices, samples are taken from `orig_data`.
    For the second part, windows are sampled from `override_data` near override events.
    """

    def __init__(
        self,
        orig_data,
        override_data,
        nlookback: int,
        nlookahead: int,
        label_length: int,
        input_sampling: int = 1,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.orig_data = orig_data
        self.override_data = override_data
        self.transform = transform
        self.nlookback = nlookback
        self.nlookahead = nlookahead
        self.label_length = label_length  # decoder lookback length
        self.input_sampling = input_sampling

        # "Virtual" length: conceptually we treat this as orig + override samples
        self.virtual_len = 2 * len(orig_data)

        # Detect override ranges in override_data (where Override is high)
        self.overrides = self._find_trajectory_override_ranges(
            self.override_data["Override"].values,
            self.override_data["TargetYaw"].values,
        )

        # Detect cycle starts in original trajectories
        self.cycle_start_idxs = self._find_cycle_starts(
            self.orig_data["TargetYaw"].values
        )

        # Align override segments to cycle starts in original data
        self.overrides2 = self._adjust_override_start_idx(
            self.overrides, self.cycle_start_idxs
        )

        # Optional global time offset (e.g., randomized per epoch)
        self.time_offset = 0

    def _adjust_override_start_idx(
        self, overrides: Iterable[Tuple[int, int, int]], cycle_start_idxs: List[int]
    ) -> List[Tuple[int, int, int]]:
        """
        Align override segments with corresponding original cycles,
        so that time marks can be mapped consistently.
        """
        j = 0
        adjusted_overrides: List[Tuple[int, int, int]] = []
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

    def _find_cycle_starts(self, arr: np.ndarray) -> List[int]:
        """
        Heuristic to find start indices of cycles in the TargetYaw signal.

        It looks for minima and tracks the max value between them. A new
        "overall cycle" (of multiple shorter cycles) is detected when the per-cycle
        max decreases, indicating the start of a larger pattern.
        """
        cycle_start_val = np.min(arr)
        max_val = 0.0
        cycle_start_idx = -1
        start_idxs_and_max: List[Tuple[int, float]] = []
        min_seconds = 5 * 100  # assuming 100 Hz sampling

        for i in range(len(arr)):
            if abs(arr[i] - cycle_start_val) < 1e-5:
                dist_to_last_cycle_start = i - cycle_start_idx
                if dist_to_last_cycle_start > min_seconds:
                    start_idxs_and_max.append((cycle_start_idx, max_val))
                    max_val = 0.0
                cycle_start_idx = i
            if arr[i] > max_val:
                max_val = arr[i]

        prev_max_per_cycle = 0.0
        first_cycles: List[int] = []
        for start_idx, max_per_cycle in start_idxs_and_max:
            if max_per_cycle < prev_max_per_cycle:
                # New overall cycle (set of cycles) starts at lower max
                first_cycles.append(start_idx)
            prev_max_per_cycle = max_per_cycle

        return first_cycles

    def _find_trajectory_override_ranges(
        self, overrides: np.ndarray, targetyaw: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Find contiguous ranges where override is active and associate each range
        with the start index of the previous cycle in TargetYaw.
        """
        cycle_start_val = np.min(targetyaw)
        cycle_start_idx = 0
        last_cycle_start_idx = 0

        override_val = np.max(overrides)
        found_len = 0
        found_len_start = 0
        override_ranges: List[Tuple[int, int, int]] = []

        # Find first override-ish start (not used later, but kept for compatibility)
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

    def __len__(self) -> int:
        # Effective length after accounting for lookback/lookahead margins
        return (
            self.virtual_len
            - 2 * self.nlookback * self.input_sampling
            - 2 * self.nlookahead
        )

    def _find_next_override(
        self,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Map a virtual index offset into an override window in override_data.
        Returns:
            seq_lookback, seq_lookahead, orig_data_start_idx
        """
        # Map offset from orig_data length scale to override_data length scale
        offset = int(offset / len(self.orig_data) * len(self.override_data))

        override_start_idx = self.overrides[0][0]  # fallback
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
            slice_start:split_idx:self.input_sampling, :
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

    def __getitem__(self, idx: int):
        """
        Return:
            seq_lookback      : encoder input window
            seq_lookahead     : decoder target window (label_len + horizon)
            seq_lookback_mark : time indices for encoder
            seq_lookahead_mark: time indices for decoder
        """
        # First part of the virtual range → original data
        if (
            idx
            < len(self.orig_data)
            - self.nlookback * self.input_sampling
            - self.nlookahead
        ):
            split_idx = idx + self.nlookback * self.input_sampling
            seq_lookback = self.orig_data.iloc[
                idx:split_idx:self.input_sampling, :
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
            # Second part of the virtual range → override data
            idx -= (
                len(self.orig_data)
                - self.nlookback * self.input_sampling
                - self.nlookahead
            )
            seq_lookback, seq_lookahead, idx = self._find_next_override(idx)

        # Optionally transform input windows (e.g., normalization)
        if self.transform:
            # Note: this may be a performance bottleneck; could be precomputed
            seq_lookback = self.transform(seq_lookback)
            seq_lookahead = self.transform(seq_lookahead)

        lookback_actual = self.nlookback * self.input_sampling

        # Simple time indices as "marks" for encoder and decoder
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

        seq_lookback = torch.tensor(seq_lookback, dtype=torch.float32)
        seq_lookahead = torch.tensor(seq_lookahead, dtype=torch.float32)

        return seq_lookback, seq_lookahead, seq_lookback_mark, seq_lookahead_mark


# ---------------------------------------------------------------------------
# Learning rate schedule (from TFB)
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    """
    Adjust optimizer learning rate according to args.lradj schedule.

    Supported types:
        - 'type1', 'type2', 'type3'
        - 'constant'
        - 'cosine'
    """
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
    elif args.lradj == "cosine":
        total_epochs = getattr(args, "num_epochs", 20)
        lr = args.lr * (0.5 * (1 + math.cos(math.pi * epoch / total_epochs)))
        lr_adjust = {epoch: lr}
    else:
        print(f"[Warning] Unknown lradj '{args.lradj}', keeping current LR.")
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def forecast_fit(model: ForecastingModel, train_dataset, validate_dataset, **kwargs) -> ForecastingModel:
    """
    Train (or fine-tune) the given model on train/validation datasets.

    Handles:
        - optional checkpoint restore
        - attaching transforms
        - DataLoader setup with deterministic seeding
        - optimizer and loss
        - epoch loop (train + validate)
        - early stopping
        - learning rate schedule
    """

    if model.model is None:
        raise ValueError("Model not initialized. Call the model initializer first.")
    if kwargs.get("use_checkpoint", True) and hasattr(model, "early_stopping"):
        if model.early_stopping.check_point is not None:
            # Local alias with explicit nn.Module type for linters
            nn_model = cast(nn.Module, model.model)
            nn_model.load_state_dict(model.early_stopping.check_point)
        else:
            nn_model = cast(nn.Module, model.model)
    else:
        nn_model = cast(nn.Module, model.model)

    # Attach transform to datasets if needed
    if not kwargs.get("already_transformed", False) and getattr(model.config, "norm", False):
        train_dataset.transform = model.scaler.transform
        validate_dataset.transform = model.scaler.transform

    config = model.config

    num_workers = getattr(config, "num_workers", 0) or 0

    # Optional deterministic seeding for workers and shuffling
    base_seed = kwargs.get("seed", None)
    if base_seed is not None:
        base_seed = cast(int, base_seed)

    def _worker_init_fn(worker_id: int) -> None:
        if base_seed is None:
            return
        seed = base_seed + worker_id
        import random as _random
        import numpy as _np
        import torch as _torch
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    data_generator: Optional[torch.Generator] = None
    if base_seed is not None:
        try:
            data_generator = torch.Generator()
            data_generator.manual_seed(base_seed)
        except Exception:
            data_generator = None

    train_data_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
        generator=data_generator,
    )
    valid_data_loader = DataLoader(
        validate_dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
        generator=data_generator,
    )

    # Loss selection
    if config.loss == "MSE":
        criterion = nn.MSELoss()
    elif config.loss == "MAE":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=0.5)

    # Optimizer selection
    opt_name = getattr(config, "optimizer", "adam").lower()
    weight_decay = getattr(config, "weight_decay", 0.0)

    if opt_name == "adam":
        optimizer = optim.Adam(nn_model.parameters(), lr=config.lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(nn_model.parameters(), lr=config.lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            nn_model.parameters(), lr=config.lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.early_stopping = common.EarlyStopping(patience=config.patience)
    nn_model.to(device)

    total_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    for epoch in range(config.num_epochs):
        train_loss_sum = 0.0
        num_batches = 0
        last_grad_norm: Optional[float] = None

        # Randomize global time offset for datasets (shifts time marks)
        if kwargs.get("random_time_offset_per_epoch", False):
            time_offset = int(np.random.randint(2**14, size=1)[0])
            train_data_loader.dataset.time_offset = time_offset  # type: ignore
            valid_data_loader.dataset.time_offset = time_offset  # type: ignore

        nn_model.train()
        for i, (input, target, input_mark, target_mark) in enumerate(train_data_loader):
            optimizer.zero_grad()

            input = input.to(device)
            target = target.to(device)
            input_mark = input_mark.to(device)
            target_mark = target_mark.to(device)

            loss_importance: Optional[torch.Tensor] = None  # reset every batch

            # Forward pass branch depending on model config
            if config.decoder_input_required:
                dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(device)
                )
                output = nn_model(input, input_mark, dec_input, target_mark)
            elif config.has_loss_importance:
                output, loss_importance = nn_model(input)
            else:
                output = nn_model(input)

            # Focus loss on forecast horizon
            target_slice = target[:, -config.horizon :, :]
            output_slice = output[:, -config.horizon :, :]
            loss = criterion(output_slice, target_slice)

            if getattr(config, "has_loss_importance", False):
                loss = loss + loss_importance  # type: ignore[name-defined]

            loss.backward()

            # Gradient norm for logging
            try:
                total_norm_sq = 0.0
                for p in nn_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm_sq += param_norm * param_norm
                last_grad_norm = total_norm_sq ** 0.5
            except Exception:
                last_grad_norm = None

            # Gradient clipping (if configured)
            if hasattr(config, "grad_clip") and config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), config.grad_clip)

            optimizer.step()

            train_loss_sum += float(loss.item())
            num_batches += 1

        # Validation
        series_dim = getattr(config, "c_out", None)
        if series_dim is None:
            # Fallback: infer from first batch: target shape [B, T, D]
            first_batch = next(iter(valid_data_loader))
            series_dim = first_batch[1].shape[-1]

        try:
            valid_loss = model.validate(valid_data_loader, series_dim, criterion)
        except TypeError:
            # For legacy models: validate(valid_data_loader, criterion)
            valid_loss = model.validate(valid_data_loader, criterion)  # type: ignore

        # Optional training progress callback
        ProgressCallback = Callable[
            [float, float, ForecastingModel, int, Dict[str, Any]],
            None
        ]

        raw_callback = kwargs.get("training_progress_callback", None)
        callback_fcn: Optional[ProgressCallback] = (
            raw_callback if callable(raw_callback) else None
        )

        if callback_fcn:
            avg_train_loss = train_loss_sum / max(1, num_batches)
            current_lr = optimizer.param_groups[0].get("lr", None)
            callback_fcn(
                valid_loss,
                model.early_stopping.val_loss_min,
                model,
                epoch,
                {
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                    "grad_norm": last_grad_norm,
                },
            )

        if model.early_stopping(valid_loss, model.model):
            break

        adjust_learning_rate(optimizer, epoch + 1, config)

    return model


# ---------------------------------------------------------------------------
# Orchestration: end-to-end training
# ---------------------------------------------------------------------------

def train_model(
    model_config: Dict[str, Any],
    evaluation_config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    High-level training entry point.

    Steps:
        1. Apply global seed from evaluation_config (if present)
        2. Build model from model_config
        3. Load & split datasets (normal + overrides)
        4. Scale data and create CustomDatasetWithOverrides instances
        5. Run forecast_fit() training loop
        6. Return validation loss, training time, and model size (in M parameters)
    """

    # Apply seed from evaluation config; prefer strategy_args.seed
    seed = None
    if isinstance(evaluation_config, dict):
        strat_args = evaluation_config.get("strategy_args", {})
        seed = strat_args.get("seed", evaluation_config.get("seed"))
    if seed is not None:
        common.set_fixed_seed(seed)

    # Instantiate the model via TFB factory
    model_factory = get_models(model_config)[0]
    raw_model = model_factory()
    model: ForecastingModel = cast(ForecastingModel, raw_model)

    # Push config params from model_config into model.config
    for key, val in model_config["models"][0].items():
        setattr(model.config, key, val)

    # Debug: print config contents
    for key, val in model.config.__dict__.items():
        print(key, val)

    # Resolve dataset paths with env/kwargs defaults
    data_path, data_path_overrides = common.resolve_data_paths(
        kwargs.get("data_path"), kwargs.get("data_path_overrides")
    )

    data = common.load_csv(data_path)
    data_w_overrides = common.load_csv(data_path_overrides)

    # Task name for TFB internals
    setattr(model.config, "task_name", "short_term_forecast")

    # Let the model perform any internal hyper-parameter logic
    model.multi_forecasting_hyper_param_tune(data)

    # Initialize underlying PyTorch model
    if hasattr(model, "_init_model"):
        model.model = model._init_model()
    else:
        model.model = model.model_class(model.config)

    lookback = model_config["models"][0]["model_hyper_params"]["seq_len"]
    horizon = evaluation_config["strategy_args"]["horizon"]
    lookahead = horizon

    train_ratio_in_tv = evaluation_config["strategy_args"]["train_ratio_in_tv"]
    tv_ratio = evaluation_config["strategy_args"]["tv_ratio"]  # train+val fraction

    # Train/val split (ignore test here)
    train_data, valid_data, _ = common.split_data(data, tv_ratio, train_ratio_in_tv)
    train_data_w_overrides, valid_data_w_overrides, _ = common.split_data(
        data_w_overrides, tv_ratio, train_ratio_in_tv
    )

    # Fit scaler on both normal and override train data
    model.scaler.partial_fit(train_data.values)
    model.scaler.partial_fit(train_data_w_overrides.values)

    # Scale copies for training and validation
    train_data = common.scaled_dataframe_copy(train_data, model.scaler)
    valid_data = common.scaled_dataframe_copy(valid_data, model.scaler)
    train_data_w_overrides = common.scaled_dataframe_copy(
        train_data_w_overrides, model.scaler
    )
    valid_data_w_overrides = common.scaled_dataframe_copy(
        valid_data_w_overrides, model.scaler
    )

    gc.collect()

    # Build datasets
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

    start_time = time.time()
    forecast_fit(
        model,
        train_dataset,
        validate_dataset,
        use_checkpoint=False,
        random_time_offset_per_epoch=True,
        already_transformed=True,
        training_progress_callback=kwargs.get("training_progress_callback", None),
        seed=seed,
    )
    elapsed = time.time() - start_time

    try:
        params_millions = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        params_millions = None

    result = {
        "val_loss": getattr(model.early_stopping, "val_loss_min", float("inf")),
        "training_time": elapsed,
        "model_params_millions": params_millions,
    }
    return result