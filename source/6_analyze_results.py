import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "../data/results/results.csv"
results = pd.read_csv(file_path)

# Reorder the agents
# TODO: Need to add approximate_q_learning to the list
results["agent_name"] = results["agent_name"].astype("category")
results["agent_name"] = results["agent_name"].cat.reorder_categories(
    ["sarsa", "q_learning", "approximate_q_learning", "deep_q_learning"])

# Rename the agents
agent_name_map ={
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
    "approximate_q_learning": "Approximate Q-Learning",
    "deep_q_learning": "Deep Q-Learning"}
results["agent_name"] = results["agent_name"].map(agent_name_map)

results["treatment_name"] = results.apply(lambda row: f"{row['agent_name']}\n({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: You can change the filters to analyze different results
# results = results[results["agent_name"] == "deep_q_learning"]
# results = results[results["curriculum"] == False]

results = results.sort_values(by=["agent_name", "curriculum"])

# NOTE: Cast curriculum to a string, so it can be used to set the hue
results["curriculum"] = results["curriculum"].astype(str)


# Set the color to blue for the baseline and orange for the curriculum
blue = sns.color_palette("tab10")[0]
orange = sns.color_palette("tab10")[1]

# Plot the total reward by agent
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="treatment_name",
    y="total_reward",
    palette=[blue, orange],
    data=results,
    capsize=0.1,
    ci=None)
plt.title("Total Reward by Agent")
plt.title("Average of Total Reward per Episode by Agent")
plt.xlabel("Agent")
plt.xticks(rotation=15, ha="right")
plt.ylabel("Total Reward")
plt.ylim(0, 1250)
ax.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))
plt.subplots_adjust(bottom=0.15)
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{p.get_height():,.0f}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 10),
        textcoords="offset points")
plt.tight_layout()
plt.savefig(f"../data/plots/results/total_reward_by_agent.png")
plt.show()

# Plot percentage of visited states by training step
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="treatment_name",
    y="percent_states_visited",
    palette=[blue, orange],
    data=results,
    ci=None)
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{p.get_height():,.0f}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 10),
        textcoords="offset points")
plt.title("Average Percentage of States Visited per Episode")
plt.xlabel("Episode")
plt.xticks(rotation=20, ha="right")
plt.ylabel("States Visited (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(f"../data/plots/results/percent_states_visited_by_agent.png")
plt.show()


# Convert duration to milliseconds
results["duration"] = results["duration"] * 1000

# Plot average episode runtime
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="treatment_name",
    y="duration",
    data=results,
    palette=[blue, orange],
    ci=None)
plt.title("Average Step Runtime by Agent")
plt.xlabel("Agent")
plt.xticks(rotation=20, ha="right")
plt.ylabel("Runtime (ms)")
plt.ylim(0, 1.5)
plt.subplots_adjust(bottom=0.15)
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{p.get_height():,.2f}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 10),
        textcoords="offset points")
plt.tight_layout()
plt.savefig(f"../data/plots/results/avg_runtime_by_agent.png")
plt.show()

# Plot the distribution of total reward by agent
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=results,
    x="total_reward",
    hue="treatment_name",
    fill=True,
    common_norm=False,
    palette="tab10",
    alpha=0.5)
plt.title("Distribution of Total Rewards by Agent and Treatment")
plt.xlabel("Total Reward")
plt.ylabel("Density")
plt.gca().get_legend().set_title("Agent / Treatment")
plt.tight_layout()
plt.savefig(f"../data/plots/results/distribution_of_total_reward_by_agent.png")
plt.show()

# Initialize a DataFrame to store results
agents = ["SARSA", "Q-Learning", "Approximate Q-Learning", "Deep Q-Learning"]
comparison_results = pd.DataFrame(columns=["Agent", "t-Statistic", "p-Value"])
for agent in agents:
    baseline_data = results[(results["agent_name"] == agent) & (results["curriculum"] == "False")]["total_reward"]
    curriculum_data = results[(results["agent_name"] == agent) & (results["curriculum"] == "True")]["total_reward"]
    t_stat, p_value = stats.ttest_rel(baseline_data, curriculum_data)
    comparison_results = comparison_results._append({
        "Agent": agent,
        "t-Statistic": t_stat,
        "p-Value": p_value},
        ignore_index=True)

# Display the results
print(comparison_results)

# Create a summary table by agent and treatment
summary = results.groupby(["agent_name", "curriculum"]).agg({
    "total_reward": ["mean", "std"],
    "percent_states_visited": ["mean", "std"],
    "duration": ["mean", "std"]}).reset_index()
summary.columns = ["Agent", "Curriculum", "Mean Total Reward", "Std Total Reward", "Mean States Visited", "Std States Visited", "Mean Duration", "Std Duration"]
print(summary)

