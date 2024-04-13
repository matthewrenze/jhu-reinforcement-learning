import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

treatments = [
    #{"agent_name": "sarsa", "use_curriculum": False, "treatment_name": "SARSA (baseline)"},
    # {"agent_name": "sarsa", "use_curriculum": True, "treatment_name": "SARSA (curriculum)"},
    #{"agent_name": "q_learning", "use_curriculum": False, "treatment_name": "Q-Learning (baseline)"},
    # {"agent_name": "q_learning", "use_curriculum": True, "treatment_name": "Q-Learning (curriculum)"},
    {"agent_name": "approximate_q_learning", "use_curriculum": False, "treatment_name": "Approximate Q-Learning (baseline)"},
    # {"agent_name": "approximate_q_learning", "use_curriculum": True, "treatment_name": "Approximate Q-Learning (curriculum)"},
    #{"agent_name": "deep_q_learning", "use_curriculum": False, "treatment_name": "Deep Q-Learning (baseline)"},
    # {"agent_name": "deep_q_learning", "use_curriculum": True, "treatment_name": "Deep Q-Learning (curriculum)"}
]

for treatment in treatments:

    agent_name = treatment["agent_name"]
    use_curriculum = False
    treatment_name = treatment["treatment_name"]
    file_name_postfix = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
    file_path = f"../data/details/hyperparameter_details_{file_name_postfix}.csv"
    details = pd.read_csv(file_path)

    details["treatment_name"] = details.apply(lambda row: f"α={row['alpha']}; γ={row['gamma']}; ε={row['epsilon']}; features={row['features']}", axis=1)

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
        .groupby(["alpha", "gamma", "epsilon"], as_index=False) \
        .apply(rolling_window, window_size)
    smoothed_details.reset_index(drop=True, inplace=True)

    details = details.sort_values(by=["alpha", "gamma", "epsilon"])

    # Plot total reward by training step grouped by hyperparameter treatment
    plot_title = f"Hyperparameter Curves for {treatment_name}"
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        x="training_step",
        y="rolling_total_reward",
        hue="treatment_name",
        data=smoothed_details)
    plt.title(plot_title)
    plt.xlabel("Training Step")
    plt.ylabel("Total Reward")
    plt.ylim(0, 500)
    plt.legend(title="Hyperparameters")
    plt.savefig(f"../data/plots/hyperparameter_curves_{file_name_postfix}.png")
    plt.show()