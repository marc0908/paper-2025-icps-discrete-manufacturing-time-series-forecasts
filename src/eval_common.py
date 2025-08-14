import argparse
import math
import os
import sys
import time

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib import pyplot
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch

import common


units_dict = {
    "Voltage0": "Volt",
    "Voltage1": "Volt",
    "PitchDot": "rad/s",
    "YawDot": "rad/s",
    "Pitch": "rad",
    "Yaw": "rad",
    "TargetPitch": "rad",
    "TargetYaw": "rad",
    "Override": "bool",
}
name_dict = {
    "Voltage0": "Voltage $U_0$",
    "Voltage1": "Voltage $U_1$",
    "PitchDot": "$\\dot{\\varTheta}$",
    "YawDot": "$\\dot{\\varPsi}$",
    "Pitch": "Pitch $\\varTheta$",
    "Yaw": "Yaw $\\varPsi$",
    "TargetPitch": "Target $\\varTheta_T$",
    "TargetYaw": "Target $\\varPsi_T$",
    "Override": "Override",
}
name_est_dict = {
    "Voltage0": "Voltage $\\hat{U_0}$",
    "Voltage1": "Voltage $\\hat{U_1}$",
    "PitchDot": "$\\hat{\\dot{\\varTheta}}$",
    "YawDot": "$\\hat{\\dot{\\varPsi}}$",
    "Pitch": "Pitch $\\hat{\\varTheta}$",
    "Yaw": "Yaw $\\hat{\\varPsi}$",
    "TargetPitch": "Target $\\hat{\\varTheta_T}$",
    "TargetYaw": "Target $\\hat{\\varPsi_T}$",
    "Override": "Override",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--nruns",
        type=int,
        required=True,
        help="Number of sample runs evaluated per model",
    )
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        help="Model name and its checkpoint path",
    )

    args = parser.parse_args()

    return args


def truncate_float(n, decimals=1):
    return int(n * (10**decimals)) / 10**decimals


def ceil(n, decimals=1):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def generate_label_values(values1, values2):
    def adjust(val, mult, decimals=1):
        if mult * val < 0:
            return truncate_float(val, decimals)
        return ceil(val, decimals)

    raw_min_val = np.min((np.min(values1), np.min(values2)))  # Sorry :o
    raw_max_val = np.max((np.max(values1), np.max(values2)))
    raw_dist = np.abs(raw_max_val - raw_min_val)
    label_dist = 0
    decimals = 1
    label_min_val = 0
    label_max_val = 0
    while raw_dist > 1.2 * label_dist:
        label_min_val = adjust(raw_min_val, 1, decimals)
        label_max_val = adjust(raw_max_val, -1, decimals)
        label_dist = np.abs(label_max_val - label_min_val)
        decimals += 1
    return (label_min_val, label_max_val)


def plot_generated_tight(actual, generated, lookback_start_idx, lookback_len, **kwargs):
    W = 7  # 5.8    # Figure width in inches, approximately A4-width - 2*1.25in margin
    pyplot.rcParams.update(
        {
            "text.usetex": True,
            # "font.family": "Helvetica",
            "text.latex.preamble": r"\usepackage{amsmath}"
            + "\n"
            + r"\usepackage{lmodern}",
            "font.size": 8,
            "legend.fontsize": 8,
            "font.family": "lmodern",
            "figure.figsize": (W, W * 0.15),
            "lines.linewidth": 0.6,
        }
    )
    pyplot.rcParams["savefig.pad_inches"] = 0

    column_names = actual.columns.to_list()
    column_names.remove("Override")
    fig, axs = pyplot.subplots(
        1, len(column_names), sharex=False, constrained_layout=True
    )

    lookback_end_idx = lookback_start_idx + lookback_len
    generated_len = generated.shape[0]
    actual_end_idx = lookback_end_idx + generated_len

    omit_len = int(lookback_len * 0.9)
    actual = actual[lookback_start_idx + omit_len : actual_end_idx]

    col_index = list(range(0, len(column_names)))

    freq = 100  # Hz
    actual_idx = np.arange(0, actual.shape[0]) / freq

    generated_index = (
        np.arange(lookback_len - omit_len, generated_len + lookback_len - omit_len)
        / freq
    )

    for i, name, ax in zip(col_index, column_names, axs):
        ax.yaxis.set_tick_params(rotation=90)
        ax.yaxis.set_tick_params(which="minor", length=0)
        ax.yaxis.set_tick_params(labelsize="small")
        y_tick_values = generate_label_values(actual.values[:, i], generated[:, i])
        ax.set_yticks(
            ticks=y_tick_values, labels=[str(l) for l in y_tick_values], va="center"
        )
        ax.set_yticks(
            ticks=[sum(y_tick_values) / 2],
            labels=[units_dict[name]],
            rotation=90,
            minor=True,
            va="center",
        )

        xtick_vals = [0, actual.shape[0] // freq]
        ax.set_xticks(ticks=xtick_vals, labels=[str(l) + " s" for l in xtick_vals])
        if not kwargs.get("show_xlabel", True):
            ax.set_xticklabels([])

        ax.plot(actual_idx, actual.values[:, i], label=name_dict[name])  # marker='x')

        ax.plot(
            generated_index,
            generated[:, i],
            label=name_est_dict[name],
            color=kwargs.get("gen_color", "orange"),
        )

        if kwargs.get("showtitles", True):
            ax.set_title(name_dict[name])

        if kwargs.get("markOverride", False) and name == "TargetPitch":
            override_idx = np.flatnonzero(np.diff(actual["Override"].values))[0]
            pitch_range = actual["TargetPitch"].iloc[override_idx : override_idx + 10]
            pitch_min = np.min(pitch_range)
            pitch_max = np.max(pitch_range)
            height = pitch_max - pitch_min
            center = actual_idx[override_idx]
            y_plot_dist = y_tick_values[1] - y_tick_values[0]

            ellipse = patches.Ellipse(
                (center, pitch_min + height / 2),
                width=1.2,
                height=height + y_plot_dist * 0.15,
                edgecolor="red",
                facecolor="none",
                lw=0.8,
                zorder=-10,
                linestyle=":",
            )

            ax.add_patch(ellipse)

            ellipse.set_clip_box(ax.bbox)
            ellipse.set_transform(ax.transData)

    for ax in axs:
        ax.axvline(x=generated_index[0], ls="--", c="red", zorder=-1.0, lw=0.5)
        if "generated_len_per_run" in kwargs:
            every_new_generated_step = generated_index[
                :: kwargs["generated_len_per_run"]
            ]
            for xpos in every_new_generated_step:
                ax.axvline(x=xpos, ls="--", c="gray", zorder=-1.0, lw=0.5)

    if "desc" in kwargs:
        axs[0].set_ylabel(kwargs["desc"], va="center", fontsize="small")

    pyplot.show()
    fig.savefig(kwargs.get("fname", "eval_graph.pdf"), dpi=1000)


def format_mean_std_pairs(mean_val, std_val, precision=3, bold=False):
    fmt = f"{{:.{precision}f}}"
    res = f"${fmt.format(mean_val)}\\pm{fmt.format(std_val)}$"
    if bold:
        res = "\\textbf{" + res + "}"
    return res


def format_mean_std(values):
    for mean_val, std_val in zip(values.mean(axis=0), values.std(axis=0)):
        yield format_mean_std_pairs(mean_val, std_val, precision=2)


def create_mae_table_row(mae_means, mae_stds):
    row = []
    for mean_val, std_val in zip(mae_means, mae_stds):
        row.append((mean_val, std_val))
    return row


def create_mae_mse_table_row(column_names, stats_multi, stats_avgs):
    excluded_cols = ["Override"]

    mae_means, mse_means = np.mean(stats_multi, axis=0)
    mae_stds, mse_stds = np.std(stats_multi, axis=0)

    row = []
    for col_name, mae_mean, mae_std, mse_mean, mse_std in zip(
        column_names, mae_means, mae_stds, mse_means, mse_stds
    ):
        if col_name in excluded_cols:
            continue
        row.append(
            (mae_mean, mae_std, mse_mean, mse_std)
        )  # & ${mse_mean:.2f}\\pm{mse_std:.2f}$")

    avg_vals = []
    for mean_val, std_val in zip(stats_avgs.mean(axis=0), stats_avgs.std(axis=0)):
        avg_vals += [mean_val, std_val]
    row.append(avg_vals)

    return row


def create_avgs_mae_mse_table_row(stats_avgs):
    row = []
    # MAE and MSE
    avg_vals = []
    for mean_val, std_val in zip(stats_avgs.mean(axis=0), stats_avgs.std(axis=0)):
        avg_vals += [mean_val, std_val]
    row.append(avg_vals)

    return row


def print_latex_table_average_only(data, results):
    model_name_spacer = [" "]
    header = model_name_spacer + ["\\multicolumn{2}{c}{Average}\\\\"]
    tabular_start = ["\\begin{tabular}{", "r" + "c" * (len(header)), "}"]
    mae_mse_row = model_name_spacer + ["MAE"] * (len(header) - 1) + ["MSE"]
    print("".join(tabular_start))
    print(" & ".join(header))
    print(" & ".join(mae_mse_row) + "\\\\")

    sorted_results = sorted(results.items(), key=lambda item: item[-1][0])

    for model_name, value_sets in sorted_results:
        row = [model_name]
        average_mae_mean, average_mae_std, average_mse_mean, average_mse_std = (
            value_sets[-1]
        )
        row.append(format_mean_std_pairs(average_mae_mean, average_mae_std))
        row.append(format_mean_std_pairs(average_mse_mean, average_mse_std))
        print(" & ".join(row) + "\\\\")
    print("\\end{tabular}")


def print_latex_table(data, results):
    excluded_cols = ["Override"]
    model_name_space = [" "]
    used_col_names = [
        name_dict[col_name]
        for col_name in list(data.columns)
        if col_name not in excluded_cols
    ]
    header = model_name_space + used_col_names
    header.append("\\multicolumn{2}{c}{Average}\\\\")
    tabular_start = "\\begin{tabular}{", "c" * (len(header) + 1), "}"
    mae_mse_row = (
        " & ".join(model_name_space + ["MAE"] * (len(header) - 1) + ["MSE"]) + "\\\\"
    )
    print("".join(tabular_start))
    print("&".join(header))
    print(mae_mse_row)

    sorted_results = sorted(results.items(), key=lambda item: item[-1][0])
    col_lowest_values = [1e10] * (len(used_col_names) + 2)
    for model_name, value_sets in sorted_results:
        for i, value_set in enumerate(value_sets):
            if col_lowest_values[i] > value_set[0]:
                col_lowest_values[i] = value_set[0]
        average_mae_mean, average_mae_std, average_mse_mean, average_mse_std = (
            value_sets[-1]
        )
        if col_lowest_values[-1] > average_mse_mean:
            col_lowest_values[-1] = average_mse_mean

    for model_name, value_sets in sorted_results:
        row = [model_name]
        for i, value_set in enumerate(value_sets):
            is_lowest = value_set[0] == col_lowest_values[i]
            row.append(
                format_mean_std_pairs(
                    value_set[0], value_set[1], precision=3, bold=is_lowest
                )
            )

        average_mae_mean, average_mae_std, average_mse_mean, average_mse_std = (
            value_sets[-1]
        )
        is_lowest = average_mse_mean == col_lowest_values[-1]
        row.append(
            format_mean_std_pairs(
                average_mse_mean, average_mse_std, precision=3, bold=is_lowest
            )
        )
        print(" & ".join(row) + "\\\\")
    print("\\end{tabular}")


def forecast_and_stats(model, data, n_recursion):
    lookback_len = model.config.seq_len
    lookback_start_idx = 0

    generated_len_per_run = model.config.horizon
    initial_lookback = data[:lookback_len]
    lookback = initial_lookback.copy()
    time_per_generation = []
    generated_results = []

    stats = {"per_variable": [], "avg": []}

    for i in range(n_recursion):
        start_time = time.time()
        generated = common.forecast_custom(
            model, lookback
        )  # TODO: for downsampling, pass dec input with full sampling rate
        time_per_generation.append(time.time() - start_time)
        lookback = np.vstack((lookback[len(generated) :, :], generated))
        generated_results.append(generated)

        reference_data_start_idx = lookback_len + i * generated_len_per_run
        reference_data_end_idx = reference_data_start_idx + generated_len_per_run
        reference_data = data[reference_data_start_idx:reference_data_end_idx]
        scaled_reference = model.scaler.transform(reference_data)
        scaled_generated = model.scaler.transform(generated)

        mae = metrics.mean_absolute_error(
            scaled_reference, scaled_generated, multioutput="raw_values"
        )
        mse = metrics.mean_squared_error(
            scaled_reference, scaled_generated, multioutput="raw_values"
        )
        stats["per_variable"].append((mae, mse))

        mae = metrics.mean_absolute_error(scaled_reference, scaled_generated)
        mse = metrics.mean_squared_error(scaled_reference, scaled_generated)
        stats["avg"].append((mae, mse))

    generated = np.vstack(generated_results)
    return stats, generated
