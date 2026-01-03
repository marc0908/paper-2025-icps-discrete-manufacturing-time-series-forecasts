import numpy as np
import torch
import matplotlib.pyplot as pyplot

from . import eval_common
from ..core import common, config
from .eval_common import forecast_and_stats, create_mae_table_row, parse_args, print_latex_table_recursive


def eval_model(modelname, modelpath, data, n_runs=10000):
    model = common.restore_model(modelpath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()

    # Load best checkpoint ONCE (if present)
    if hasattr(model, "early_stopping") and model.early_stopping.check_point is not None:
        model.model.load_state_dict(model.early_stopping.check_point)

    # store device for reuse (optional but clean)
    model._eval_device = device

    print("========= ", modelname)
    print("Model scaler (must match):", model.scaler)
    print("Model parameter count: ", common.sum_model_params(model))

    lookback_len = model.config.seq_len
    generated_len_per_run = model.config.horizon

    n_recursion = 100 + 1

    cfg = config.default_eval_config()
    seed = cfg["strategy_args"]["seed"]
    common.set_fixed_seed(seed)
    rng = np.random.default_rng(seed=seed)
    print("Seed:", seed)

    train_data, valid_data, test_data = common.split_data(
        data,
        cfg["strategy_args"]["tv_ratio"],
        cfg["strategy_args"]["train_ratio_in_tv"],
    )

    trajectory_starts = list(
        rng.integers(
            low=0,
            high=len(test_data) - lookback_len - n_recursion * generated_len_per_run,
            size=n_runs,
        )
    )

    stats_avgs = []
    stats_per_variable = []
    generated_results = []

    for trajectory_start_idx in trajectory_starts:
        in_data = test_data[trajectory_start_idx:].values
        stat, generated = forecast_and_stats(model, in_data, n_recursion)
        generated_results.append(generated)
        stats_avgs.append(stat["avg"])
        stats_per_variable.append(stat["per_variable"])

    stats_avgs = np.dstack(stats_avgs)

    # stats_avgs[0] ... all first results of every run (up to nrecursion-1 results)
    # stats_avgs[0,0] ... all MAE of every first recursion
    # stats_avgs[0,1] ... all MSE of every first recursion
    means_stats_avgs = np.mean(stats_avgs, axis=2)

    std_stats_avgs = np.std(stats_avgs, axis=2)
    mae_means = means_stats_avgs[:, 0]
    mae_stds = std_stats_avgs[:, 0]

    mae_vals = stats_avgs[:, 0, :]
    mae_vals_sorted = np.sort(mae_vals, axis=1)
    low_quart_runs = mae_vals_sorted[:, n_runs // 4]
    median_runs = mae_vals_sorted[:, n_runs // 2]
    up_quart_runs = mae_vals_sorted[:, (3 * n_runs) // 4]

    return create_mae_table_row(mae_means, mae_stds), (
        low_quart_runs,
        median_runs,
        up_quart_runs,
    )


def plot_stats(stats):
    W = 4  # Figure width in inches, approximately A4-width - 2*1.25in margin
    pyplot.rcParams.update(
        {
            "text.usetex": eval_common.USE_LATEX,
            "text.latex.preamble": r"\usepackage{amsmath}"
            + "\n"
            + r"\usepackage{lmodern}",
            "font.size": 8,
            "legend.fontsize": 7,
            "font.family": "lmodern",
            "figure.figsize": (W, W * 0.45),
            "lines.linewidth": 0.6,
        }
    )
    pyplot.rcParams["savefig.pad_inches"] = 0

    fig, ax = pyplot.subplots(1, 1, sharex=True, constrained_layout=True)

    for model_name, model_stats in stats.items():
        low_quart_runs, median_runs, up_quart_runs = model_stats
        ar1 = ax.plot(median_runs, "-", label=model_name)
        n_steps = len(up_quart_runs)
        ar2 = ax.fill_between(
            np.arange(n_steps), low_quart_runs, up_quart_runs, alpha=0.2
        )
        pyplot.locator_params(axis="x", integer=True, tight=True)

    import matplotlib.lines as mlines

    note = mlines.Line2D([], [], label="*) inidcates second training pass", alpha=0.0)
    handles, labels = pyplot.gca().get_legend_handles_labels()
    handles.append(note)
    labels.append("\n*) indicates\nthe second\ntraining pass")
    ax.legend(
        handles=handles, labels=labels, ncols=1, bbox_to_anchor=(1, 1), loc="upper left"
    )

    ax.set_xlabel("Recursion Step")
    ax.set_ylabel("Overall MAE")
    ax.set_yscale("log")

    pyplot.show()
    fig.savefig("stats_v2.pdf", dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    data = common.load_csv(args.data_path)

    results = {}
    results_med_quarts = {}

    for model_name, model_path in args.model:
        result_row, median_w_quarts = eval_model(
            model_name, model_path, data, n_runs=args.nruns
        )
        results[model_name] = result_row
        results_med_quarts[model_name] = median_w_quarts

    print_latex_table_recursive(data, results, stepsize=1)
    print_latex_table_recursive(data, results, stepsize=20)
    plot_stats(results_med_quarts)
