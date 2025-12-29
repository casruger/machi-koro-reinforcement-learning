import time
import numpy as np
import torch  # Required to inspect the neural network internals
from stable_baselines3 import PPO
from machi_gym import MachiKoroEnv
from ai_players import HumanPlayer
from machi_koro import ALL_CARDS, Game

# Helper function to display AI decision probabilities
def print_ai_thinking(model, obs):
    # 1. Convert observation to a format PyTorch understands
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    
    # 2. Retrieve the distribution (the AI's probability calculation)
    distribution = model.policy.get_distribution(obs_tensor)
    
    # 3. Extract raw probabilities
    probs = distribution.distribution.probs.detach().numpy()[0]
    
    # 4. Map probabilities to card names
    # Action 0 = PASS, Action 1..19 = Cards in ALL_CARDS order
    options = []
    
    # Option 0: PASS
    options.append((probs[0], "PASS (Buy nothing)"))
    
    # Options 1 to 19: Cards
    for i, card_class in enumerate(ALL_CARDS):
        p = probs[i + 1]
        card_name = card_class.__name__
        options.append((p, card_name))
    
    # 5. Sort from highest to lowest probability
    options.sort(key=lambda x: x[0], reverse=True)
    
    # 6. Print the Top 5 choices
    print("\n🧠 AI Reasoning:")
    for i in range(5):
        percentage = options[i][0] * 100
        name = options[i][1]
        if percentage > 0.1: # Only show if probability > 0.1%
            bar = "█" * int(percentage / 5)
            print(f"   {percentage:5.1f}% : {name} {bar}")
    print("")


# 1. Initialize the environment
env = MachiKoroEnv(verbose=True) # Set verbose to True to see live updates

# 2. Load the model
try:
    model = PPO.load("ppo_machi_koro")
except:
    print("No model found, using an untrained (dummy) AI.")
    model = PPO("MlpPolicy", env)

# 3. Setup Human vs AI game
obs, _ = env.reset()
env.opponent = HumanPlayer()
env.game = Game(env.agent, env.opponent)
env._play_pre_buy_phase(env.agent, is_main_agent=True)
obs = env._get_obs()

done = False
print("\n" + "="*40)
print(" WELCOME TO MACHI KORO: AI ANALYSIS MODE")
print("="*40 + "\n")

while not done:
    # Display AI reasoning
    print_ai_thinking(model, obs) # <-- Comment out this line to hide AI thinking process
    
    # Execute AI action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    # Check for game completion
    done = terminated or truncated
    if not done:
        print("-" * 20)

if reward > 0:
    print("\n🏆 THE AI HAS WON!")
else:
    print("\n🎉 YOU HAVE WON!")