from typing import Iterable, List, Optional, Tuple, Callable
from ..tslib.layers.Embed import FREQ_MAP # Use tslib's frequency map

import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDatasetWithOverrides(Dataset):
    """
    Dataset that mixes 'normal' trajectories and 'override' trajectories.

    It produces encoder/decoder input windows and corresponding "time marks"
    for models like DUET that use positional/time encodings as a separate input.

    For the first part of indices, samples are taken from `orig_data`.
    For the second part, windows are sampled from `override_data` near override events.

    About timemarks:
    The dimension of the timemarks can be set via tslib's hyperparameter "freq".
    See FREQ_MAP in tslib/layers/Embed.py for details.
    If freq is anything other than 'm' meaning more than 1 dimension for the timestamp, it is
    necessary to enable generate_temporal_features.

    Parameters
    ----------
    orig_data : pandas.DataFrame
        Time series containing only normal cyclic process behaviour.
    override_data : pandas.DataFrame
        Time series containing occasional override events (Override == 1).
    nlookback : int
        Length of the encoder input window (number of past samples).
    nlookahead : int
        Prediction horizon length (number of future samples to forecast).
    label_length : int
        Number of decoder warmup samples (initial known future samples).
    input_sampling : int, default=1
        Step size for downsampling the input sequence (e.g., 1 = no downsampling).
    transform : Callable or None, default=None
        Optional preprocessing function applied to each window (e.g., scaling).
    stride_samples : int, default=1
        Step size used when sliding over the virtual index space.
    generate_temporal_features: bool, default=False
        Whether to generate temporal embedding features using sin/cos for time marks.
    freq = 'm', default='m'
        Frequency string determining the dimension of temporal embeddings.

    Returns (per item)
    ------------------
    seq_lookback : torch.Tensor
        Encoder input window of shape [nlookback, D].
    seq_lookahead : torch.Tensor
        Decoder target window of shape [label_length + nlookahead, D].
    seq_lookback_mark : torch.Tensor
        Time index markers for the encoder window.
    seq_lookahead_mark : torch.Tensor
        Time index markers for the decoder window.
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
        stride_samples: int = 1,
        generate_temporal_features: bool = False,
        freq: str | None = None,
    ):
        self.orig_data = orig_data
        self.override_data = override_data
        self.transform = transform
        self.nlookback = nlookback
        self.nlookahead = nlookahead
        self.label_length = label_length  # decoder lookback length
        self.input_sampling = input_sampling
        self.stride_samples = max(1, int(stride_samples))
        self.generate_temporal_features = generate_temporal_features
        self.freq = freq

        # "Virtual" length: conceptually we treat this as orig + override samples
        self.virtual_len = 2 * len(orig_data)

        # Detect override ranges in override_data (where Override is high)
        self.overrides = self._find_trajectory_override_ranges(
            self.override_data["Override"].values,
            self.override_data["TargetYaw"].values,
        )

        # Optional global time offset (e.g., randomized per epoch)
        self.time_offset = 0

        # precompute valid virtual start indices with stride
        max_virtual_idx = (
            self.virtual_len
            - 2 * self.nlookback * self.input_sampling
            - 2 * self.nlookahead
        )
        max_virtual_idx = max(0, max_virtual_idx)
        self._virtual_indices = np.arange(
            0, max_virtual_idx, self.stride_samples, dtype=int
        )


    def _find_trajectory_override_ranges(
        self, overrides: np.ndarray, targetyaw: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Find contiguous ranges where override is active and associate each range
        with the start index of the previous cycle in TargetYaw.

        Returns a list of tuples:
        (override_start_idx, override_end_idx, last_cycle_start_idx)

        where:
            - override_start_idx : index where an Override==1 segment begins
            - override_end_idx   : index where this segment ends
            - last_cycle_start_idx : the start index of the most recent cycle
                in the TargetYaw signal before the override began

        This information is later used to align override windows with the cycle
        structure of the original data.
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
        return len(self._virtual_indices)

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
        scale = len(self.override_data) / len(self.orig_data) # scale factor eg. /1.000.000/300.000 = ~3.33
        offset = int(offset * scale) # index in override_data

        # self.overrides format: List of (start_idx, end_idx, last_cycle_start_idx)
        override_start_idx, _, _ = self.overrides[0]

        # Find the next override whose start_idx is beyond the scaled offset
        for start_idx, end_idx, override_cycle_start_idx in self.overrides:
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

        # Extract Lookback-sequence by taking all rows from slice_start to split_idx and all columns
        seq_lookback = self.override_data.iloc[
            slice_start:split_idx:self.input_sampling, :
        ].values

        # Extract Lookahead-sequence by taking all rows 
        # from split_idx - label_length to split_idx + nlookahead and all columns
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
        return seq_lookback, seq_lookahead, slice_start

    def __getitem__(self, idx: int):
        """
        Return:
            seq_lookback      : encoder input window
            seq_lookahead     : decoder target window (label_len + horizon)
            seq_lookback_mark : time indices for encoder
            seq_lookahead_mark: time indices for decoder
        """
        # Map dataset index to "virtual" start index with stride
        base_idx = int(self._virtual_indices[idx])

        # Limit, until when we are using original data
        orig_limit = (
            len(self.orig_data)
            - self.nlookback * self.input_sampling
            - self.nlookahead
        )

        if base_idx < orig_limit:
            # Original branch
            split_idx = base_idx + self.nlookback * self.input_sampling
            seq_lookback = self.orig_data.iloc[
                base_idx:split_idx:self.input_sampling, :
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

            idx_for_marks = base_idx
        else:
            # Original branch with overrides
            override_offset = base_idx - orig_limit
            seq_lookback, seq_lookahead, idx_for_marks = self._find_next_override(
                override_offset
            )

        # Optional transform
        if self.transform:
            seq_lookback = self.transform(seq_lookback)
            seq_lookahead = self.transform(seq_lookahead)

        lookback_actual = self.nlookback * self.input_sampling

        # Time marks, original -> just withh idx_for_marks
        seq_lookback_mark = (
            torch.arange(0, lookback_actual, self.input_sampling)
            + idx_for_marks
            + self.time_offset
        ).unsqueeze(-1)
        seq_lookahead_mark = (
            torch.arange(
                lookback_actual - self.label_length, lookback_actual + self.nlookahead
            )
            + idx_for_marks
            + self.time_offset
        ).unsqueeze(-1)

        # Adapt time marks for temporal embedding if needed
        if self.generate_temporal_features:
            seq_lookback_mark, seq_lookahead_mark = self._apply_temporal_embedding(
                seq_lookback_mark, seq_lookahead_mark
            )

        # always convert to float
        seq_lookback = torch.tensor(seq_lookback, dtype=torch.float32)
        seq_lookahead = torch.tensor(seq_lookahead, dtype=torch.float32)

        seq_lookback_mark = torch.tensor(seq_lookback_mark, dtype=torch.float32)
        seq_lookahead_mark = torch.tensor(seq_lookahead_mark, dtype=torch.float32)
        
        return seq_lookback, seq_lookahead, seq_lookback_mark, seq_lookahead_mark

    def _apply_temporal_embedding(self, mark_lookback: torch.Tensor, mark_lookahead: torch.Tensor):
        F = FREQ_MAP.get(self.freq, None) if self.freq is not None else None
        if F is None:
            raise ValueError(f"Invalid Frequency '{self.freq}' use one of {list(FREQ_MAP.keys())}")

        def encode(mark):
            pos = mark.float()

            if F == 1:
                return pos

            pe_dims = F - 1
            num_pairs = (pe_dims + 1) // 2

            i = torch.arange(0, num_pairs, device=pos.device)
            div = torch.exp(-torch.log(torch.tensor(10000.0)) * (2 * i / num_pairs))

            phase = pos * div.view(1, -1)

            sin = torch.sin(phase)
            cos = torch.cos(phase)

            pe = torch.stack([sin, cos], dim=-1).reshape(pos.shape[0], -1)
            pe = pe[:, :pe_dims]

            return torch.cat([pos, pe], dim=-1)

        return encode(mark_lookback), encode(mark_lookahead)
    

def dump_dataset_to_text(dataset, file_path="dataset_dump.txt", max_samples=None):
    """
    Extremely compact debug dump:
      - Last lookback time mark
      - First lookahead time mark
      - The time jump between them
    """

    if max_samples is None:
        max_samples = len(dataset)

    with open(file_path, "w") as f:
        f.write(f"Dataset length: {len(dataset)} samples\n")
        f.write("="*60 + "\n\n")

        for i in range(min(max_samples, len(dataset))):
            _, _, lb_mark, la_mark = dataset[i]

            f.write(f"LB {lb_mark[0].item()}-{lb_mark[-1].item()} | LA {la_mark[0].item()+800}-{la_mark[-1].item()}\n")

    print(f"Dataset successfully dumped to '{file_path}'")

