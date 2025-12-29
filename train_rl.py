import os
from stable_baselines3 import PPO
from machi_gym import MachiKoroEnv

env = MachiKoroEnv()

# Filename of the pre-trained model
model_path = "ppo_machi_koro.zip" 

# Load Model or Create New
if os.path.exists(model_path):
    print(f"--- EXISTING MODEL FOUND: {model_path} ---")
    print("Resuming training from the previous session.")
    
    # Load the model and attach it to the current environment
    model = PPO.load(model_path, env=env)
else:
    print("--- NO MODEL FOUND ---")
    print("Starting a new training session from scratch.")
    model = PPO("MlpPolicy", env, verbose=1)

# Training
STAPPEN = 100000 
print(f"Starting training for {STAPPEN} steps...")
model.learn(total_timesteps=STAPPEN)
print("Training complete!")


# Overwrite the old file with the updated version
model.save("ppo_machi_koro")
print("Model saved as 'ppo_machi_koro.zip'")