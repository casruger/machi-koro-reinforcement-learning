# 🏙️ Machi Koro Reinforcement Learning Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Stable--Baselines3-green)
![Algorithm](https://img.shields.io/badge/Algorithm-PPO-orange)

This repository contains a Reinforcement Learning (RL) agent trained to play the Japanese city-building card game **Machi Koro**. The agent utilizes **Proximal Policy Optimization (PPO)** and learns optimal winning strategies through **Self-Play**.

## 🎲 What is Machi Koro?

Machi Koro is a fast-paced card game where players act as mayors building their own cities.
* **The Goal:** Be the first player to construct all four **Landmarks** (Train Station, Shopping Mall, Amusement Park, Radio Tower).
* **Gameplay Loop**
    1.  **Roll:** Players roll dice each turn.
    2.  **Earn:** Based on the dice roll, cards (establishments) activate and earn coins for players.
    3.  **Build:** Players use coins to purchase new establishments or landmarks from the marketplace.

## 🤖 The AI Agent

The agent is built using `stable-baselines3` and `gymnasium`. It treats the game as a Markov Decision Process (MDP).


### Training Methodology
* **Algorithm:** PPO (Proximal Policy Optimization) using an `MlpPolicy` (Multi-Layer Perceptron).
* **Self-Play:** The agent trains against an opponent. Initially, this opponent plays randomly. As the agent improves and saves its model, it loads previous versions of itself to act as the opponent, creating a curriculum of increasing difficulty.

### Environment Details (`machi_gym.py`)
* **Observation Space:** A `Box(59,)` array containing:
    * Player balances (Agent & Opponent).
    * Card counts in the Agent's hand.
    * Card counts in the Opponent's hand.
    * Card counts available in the Market.
* **Action Space:** A `Discrete(20)` space representing:
    * `0`: Pass (Do nothing).
    * `1-19`: Buy a specific card (from the list of 19 available card types).
* **Reward Structure:**
    * **+100**: Winning the game.
    * **-100**: Losing the game.
    * **0**: Valid move.
    * **Negative penalties**: Attempting illegal moves (e.g., buying a card without funds) results in the turn being skipped (passed).

## 📂 Project Structure

* `machi_koro.py` & `cards.py`: The core game engine and logic.
* `machi_gym.py`: The custom Gymnasium environment wrapper connecting the game to the AI.
* `train_rl.py`: Script to train the PPO agent. It handles model saving/loading for self-play.
* `play_human_vs_ai.py`: An interactive script to play against the trained bot. It includes a visualization of the AI's probability distribution (reasoning).
* `ai_players.py`: Defines the `GymPlayer`, `RandomPlayer`, and `HumanPlayer` classes.

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Set up a virtual environment:**

    * *Windows:*
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * *Mac/Linux:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.pip
    ```

## 🚀 Usage

### 1. Training the Agent
To start training the agent from scratch (or continue from a saved model):
```bash
python -m train_rl
```

* This will save the model as `ppo_machi_koro.zip`.
* If a model already exists, it will load it as the opponent for the new training session.

### 2. Playing Against the Agent
To test your skills against the AI:
```bash
python -m play_human_vs_ai
```
 By default, the script displays the probability percentages for the cards the AI considered buying.

To disable the AI “mind-reading” probabilities for a fairer game, comment out the `print_ai_thinking` call in `play_human_vs_ai.py` (around line 71).


## 📋 Requirements

- Python 3.10+
- numpy
- gymnasium
- torch
- stable-baselines3

## 🙏 Credits

Special thanks to **Elliot Penson** for the original Python implementation of Machi Koro, which served as the foundation for this project:

👉 https://github.com/ElliotPenson/embark/tree/master
