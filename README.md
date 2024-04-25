# Curriculum Learning for RL Agents in a Pac-Man Environment

**Authors:** [Matthew Renze](https://matthewrenze.com) and [Simran Shinh](https://www.linkedin.com/in/simran-shinh/)  
**Class:** EN.705.741 - Reinforcement Learning    
**Date:** 2024-05-01

## Abstract
In this research project, we trained a series of Reinforcement Learning (RL) agents to play the game of Pac-Man using curriculum learning (CL). First, we created a custom version of the Pac-Man game to simplify the environment's dynamics. Next, we trained four RL agents with and without CL. Then, we ran each treatment for 100 episodes and analyzed the results. Our analysis revealed that CL improved the performance of the SARSA and Q-learning agents but failed to improve the Approximate Q-Learning and Deep Q-learning agents.

## Presentation
[![Presentation of the research project](https://img.youtube.com/vi/zIPDyFPdlvc/0.jpg)](https://youtu.be/zIPDyFPdlvc "Click to view on YouTube")


## Documents
 - [Research Paper](documents/paper.pdf)
 - [Slides](documents/slides.pdf)

## Videos
 - [Presentation](https://youtu.be/zIPDyFPdlvc)
 - [SARSA Agent (Baseline) Training](https://youtu.be/dvnmwI5lv2c)
 - [SARSA Agent (Curriculum) Training](https://youtu.be/f4liah7wjIY)
 - [DQN Agent (Baseline) Trained](https://youtu.be/enzh485xQU4)
 - [DQN Agent (Curriculum) Trained](https://youtu.be/sqqRLXfCOGA)

## Agents
 - [SARSA](source/agents/sarsa_agent.py)
 - [Q-learning](source/agents/q_learning_agent.py)
 - [Approximate Q-learning](source/agents/approximate_q_learning_agent.py)
 - [Deep Q-learning](source/agents/deep_q_learning_agent.py)

## Scripts
 - [1a. Sweep Hyperparameters](source/1a_sweep_hyperparameters.py) - performs a sweep of the hyperparameters for a single agent
 - [1b. Sweep Features](source/1b_sweep_features.py) - performs a sweep of the features for the Approximate Q-learning (AQL) agent
 - [2a. Analyze Hyperparameters](source/2a_analyze_hyperparameters.py) - plots the learning curves for the hyperparameter sweeps
 - [2b. Analyze Features](source/2b_analyze_features.py) - plots the learning curves for the feature-tuning sweeps for the AQL agent
 - [3. Train Agents](source/3_train_agents.py) - trains each agent over 1 million steps
 - [4. Analyze Learning](source/4_analyze_learning.py) - plots the learning curves for the agents during the training process
 - [5. Evaluate Agents](source/5_evaluate_agents.py) - evaluates the performance of each agent over 100 episodes
 - [6. Analyze Results](source/6_analyze_results.py) - plots a comparison of the agents by their total reward, percent of states visited, and runtime duration
 - [7. Play Game](source/7_play_game.py) - starts a game for a specified agent in interactive mode for observation and demonstration

## Data
 - [Levels](data/levels/) - the map files for each curriculum level and the final level
 - [Hyperparameters](data/hyperparameters/) - the results of the hyperparameter sweeps for each agent
 - [Features](data/features/) - the results of the feature-tuning sweeps for the Approximate Q-learning agent
 - [Training](data/training/) - the results of the training process for each agent
 - [Models](data/models/) - the fully trained models stored in a persistent state
 - [Results](data/results/) - the results from the evaluation of each of the fully trained agents
 
 ## Plots
  - [Hyperparameters](data/plots/hyperparameters/) - the learning curves for each agent by hyperparameter combination
  - [Features](data/plots/features/) - the learning curves for the Approximate Q-learning agent by feature combination
  - [Training](data/plots/training/) - the learning curves for each agent while training over 1 million steps
  - [Results](data/plots/results/) - a comparison of the fully trained agents by their total reward, percent of states visited, and runtime duration
