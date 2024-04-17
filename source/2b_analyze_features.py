import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

treatments = [
    {"agent_name": "approximate_q_learning", "use_curriculum": False, "treatment_name": "Approximate Q-Learning (baseline)"},
    # {"agent_name": "approximate_q_learning", "use_curriculum": True, "treatment_name": "Approximate Q-Learning (curriculum)"},
]

for treatment in treatments:

    agent_name = treatment["agent_name"]
    use_curriculum = treatment["use_curriculum"]
    treatment_name = treatment["treatment_name"]
    folder_path = "../data/features"
    file_name = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}.csv"
    file_path = f"{folder_path}/{file_name}"
    details = pd.read_csv(file_path)

    details["treatment_name"] = details.apply(lambda row: f"Feature={row['features']}", axis=1)

    # NOTE: Change  the filters (below) to analyze different details
    # details = details[details["training_step"] < 100000]

    # Note: Keep only one out of every n rows to speed up visualization
    details = details[details["training_step"] % 100 == 0]

    def rolling_window(x, window_size):
        x = x.sort_values("training_step")
        x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
        return x

    window_size = 10000
    smoothed_details = details \
        .groupby(["treatment_name"], as_index=False) \
        .apply(rolling_window, window_size)
    smoothed_details.reset_index(drop=True, inplace=True)

    details = details.sort_values(by=["alpha", "gamma", "epsilon"])

    # Plot total reward by training step grouped by hyperparameter treatment
    plot_file_name = f"{file_name.split('.')[0]}.png"
    plot_title = f"Feature Learning Curves for {treatment_name}"
    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(
        x="training_step",
        y="rolling_total_reward",
        hue="treatment_name",
        data=smoothed_details)
    plt.title(plot_title)
    plt.xlabel("Training Step")
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.ylabel("Total Reward")
    plt.ylim(0, 500)
    plt.legend(title="Hyperparameters")
    plt.savefig(f"../data/plots/features/{plot_file_name}")
    plt.show()