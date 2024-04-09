import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# DEBUG: USE THIS CODE AND DELETE THE CODE BELOW
# folder_path = "../data/training"
# details = pd.DataFrame()
# for file_name in os.listdir(folder_path):
#     if not file_name.endswith(".csv"):
#         continue
#     file_path = f"{folder_path}/{file_name}"
#     partial_details = pd.read_csv(file_path)
#     details = pd.concat([details, partial_details])

# DEBUG: BEGIN REMOVE
details_baseline = pd.read_csv("../data/training/dqn_baseline_0.001.csv")
details_0_001 = pd.read_csv("../data/training/dqn_curriculum_0.001.csv")
details_0_01 = pd.read_csv("../data/training/dqn_curriculum_0.01.csv")
details_0_1 = pd.read_csv("../data/training/dqn_curriculum_0.1.csv")


details_baseline["treatment_name"] = details_baseline.apply(lambda row: f"Deep Q-Learning (baseline 0.001)", axis=1)
details_0_001["treatment_name"] = details_0_001.apply(lambda row: f"Deep Q-Learning (curriculum 0.001)", axis=1)
details_0_01["treatment_name"] = details_0_01.apply(lambda row: f"Deep Q-Learning (curriculum 0.01)", axis=1)
details_0_1["treatment_name"] = details_0_1.apply(lambda row: f"Deep Q-Learning (curriculum 0.1)", axis=1)


details = pd.concat([details_baseline, details_0_001, details_0_01, details_0_1])
details = details[details["training_step"] < 1_000_000]
# DEBUG: END REMOVE


# details["treatment_name"] = details.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: Change  the filters (below) to analyze different details
# details = details[details["training_step"] < 100_000]
# details = details[details["agent_name"] == "deep_q_learning"]
# details = details[details["curriculum"] == False]

# Note: Keep only one out of every n rows to speed up visualization
details = details[details["training_step"] % 100 == 0]

def rolling_window(x, window_size):
    x = x.sort_values("training_step")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

window_size = 10_000
smoothed_details = details \
    .groupby(["treatment_name"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_details.reset_index(drop=True, inplace=True)

details = details.sort_values(by=["agent_name", "curriculum"])

# Plot total reward by training step
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="training_step",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_details)
plt.xlabel("Training Step")
plt.ylabel("Total Reward")
#plt.ylim(0, 500)
plt.ylim(0, 1_000)
plt.title("Total Reward per Training Step")
plt.savefig(f"../data/plots/learning_curves.png")
plt.show()

test = details[details["curriculum"] == True]
