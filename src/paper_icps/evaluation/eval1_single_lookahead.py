import numpy as np

from ..core import common, config
from .eval_common import forecast_and_stats, plot_generated_tight, create_mae_mse_table_row, parse_args, print_latex_table


def eval_model(model_name, modelpath, data, n_runs=100):
    model = common.restore_model(modelpath)
    print("========= ", model_name)
    print("Model scaler (must match):", model.scaler)
    print("Model parameter count: ", common.sum_model_params(model))

    lookback_len = model.config.seq_len
    generated_len_per_run = model.config.horizon

    n_recursion = 1

    cfg = config.default_eval_config()
    seed = cfg["strategy_args"]["seed"]
    common.set_fixed_seed(seed)

    train_data, valid_data, test_data = common.split_data(
        data,
        cfg["strategy_args"]["tv_ratio"],
        cfg["strategy_args"]["train_ratio_in_tv"],
    )

    rng = np.random.default_rng(seed=seed)
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

    stats_avgs = np.vstack(stats_avgs)
    stats_per_variable = np.vstack(stats_per_variable)

    lowest_mae, lowest_mse = stats_avgs.min(axis=0)
    lowest_idx = stats_avgs.argmin(axis=0)[0]

    highest_mae, highest_mse = stats_avgs.max(axis=0)
    highest_idx = stats_avgs.argmax(axis=0)[0]

    print("Lowest MAE:", lowest_mae, "MSE:", lowest_mse, "at index:", lowest_idx)
    print("Highest MAE:", highest_mae, "MSE:", highest_mse, "at index:", highest_idx)

    plot_generated_tight(
        test_data,
        generated_results[lowest_idx],
        trajectory_starts[lowest_idx],
        lookback_len,
        generated_len_per_run=generated_len_per_run,
        fname=f"{model_name}_lowest_mae_idx.pdf",
        gen_color="lime",
        show_xlabel=False,
        desc="\\textbf{Min MAE}" + f" ({lowest_mae:.3f})",
    )
    plot_generated_tight(
        test_data,
        generated_results[highest_idx],
        trajectory_starts[highest_idx],
        lookback_len,
        generated_len_per_run=generated_len_per_run,
        fname=f"{model_name}_highest_mae_idx.pdf",
        showtitles=False,
        desc="\\textbf{Max MAE}" + f" ({highest_mae:.3f})",
    )
    return create_mae_mse_table_row(
        test_data.columns, stats_per_variable, stats_avgs
    )


if __name__ == "__main__":
    args = parse_args()

    data = common.load_csv(args.data_path)
    results = {}

    for model_name, model_path in args.model:
        result_row = eval_model(model_name, model_path, data, n_runs=args.nruns)
        results[model_name] = result_row
    print_latex_table(data, results)
