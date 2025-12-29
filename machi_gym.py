import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
from collections import Counter
from stable_baselines3 import PPO

from machi_koro import Game, ALL_CARDS
from ai_players import RandomPlayer, GymPlayer, HumanPlayer 

class MachiKoroEnv(gym.Env):
    def __init__(self, verbose=False):
        super(MachiKoroEnv, self).__init__()
        
        self.verbose = verbose 
        
        self.card_types = ALL_CARDS
        self.n_cards = len(self.card_types)
        
        self.action_space = spaces.Discrete(self.n_cards + 1)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2 + 3 * self.n_cards,), dtype=np.float32)
        
        self.game = None
        self.agent = None
        self.opponent = None
        
        self.opponent_model = None
        model_path = "ppo_machi_koro.zip"
        
        if os.path.exists(model_path):
            if self.verbose: print(f"--- OPPONENT LOADED: {model_path} ---")
            self.opponent_model = PPO.load(model_path)
        else:
            if self.verbose: print("--- NO MODEL FOUND: Opponent plays Randomly ---")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent = GymPlayer()
        self.opponent = GymPlayer()    
        
        self.game = Game(self.agent, self.opponent)
        
        self._play_pre_buy_phase(self.agent, is_main_agent=True)
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- PHASE 1: AGENT BUYS ---
        self._execute_buy_action(self.agent, action, is_main_agent=True)
        
        if self.agent.has_won():
            if self.verbose: print("🏆 THE AI HAS WON!")
            reward += 100
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        self.game.switch_player() 

        # --- PHASE 2: OPPONENT PLAYS ---
        if not terminated:
            # Opponent rolls and earns money
            self._play_pre_buy_phase(self.opponent, is_main_agent=False)
            
            # Decide purchase
            # If human player:
            if isinstance(self.opponent, HumanPlayer):
                available_opp = self.game.find_available_cards(self.opponent)
                # Code pauses here awaiting terminal input
                card_to_buy = self.opponent.construct(available_opp) 
                
                if card_to_buy:
                    self.game.purchase_card(card_to_buy, self.opponent)
                    if self.verbose: print(f"👤 YOU buy: {card_to_buy.__name__}")
                else:
                    if self.verbose: print(f"👤 YOU pass.")
            
            # If playing against loaded model
            elif self.opponent_model:
                obs_for_opp = self._get_obs_flipped()
                opp_action, _ = self.opponent_model.predict(obs_for_opp, deterministic=True)
                self._execute_buy_action(self.opponent, int(opp_action), is_main_agent=False)

            # If playing against Random player when no model is loaded
            else:
                available_opp = self.game.find_available_cards(self.opponent)
                affordable = [c for c in available_opp if self.opponent.has_funds_for(c(self.opponent, self.game))]
                if affordable:
                    choice = random.choice(affordable)
                    opp_action = self.card_types.index(choice) + 1
                    self._execute_buy_action(self.opponent, opp_action, is_main_agent=False)

            if self.opponent.has_won():
                if self.verbose: print("🎉 YOU HAVE WON!")
                reward -= 100 
                terminated = True
                return self._get_obs(), reward, terminated, truncated, info

            self.game.switch_player()

        # --- PHASE 3: AGENT PREPARATION FOR NEXT TURN ---
        if not terminated:
            self._play_pre_buy_phase(self.agent, is_main_agent=True)

        return self._get_obs(), reward, terminated, truncated, info

    def _execute_buy_action(self, player, action, is_main_agent):
        player_name = "🤖 AI" if is_main_agent else "👤 OPPONENT"
        
        if action > 0:
            card_idx = action - 1
            if card_idx < len(self.card_types):
                card_class = self.card_types[card_idx]
                available = self.game.find_available_cards(player)
                dummy_card = card_class(player, self.game)
                
                if card_class in available and player.has_funds_for(dummy_card):
                    self.game.purchase_card(card_class, player)
                    if self.verbose: print(f"{player_name} buys: {card_class.__name__}")
                else:
                    # AI attempted illegal move (too expensive or out of stock)
                    pass
            else:
                # Invalid card index out of bounds
                pass
        else:
            # If action == 0, the player passes
             if self.verbose: print(f"{player_name} decides to PASS.")

    def _play_pre_buy_phase(self, player, is_main_agent):
        player_name = "🤖 AI" if is_main_agent else "👤 YOU"
        
        # Track balance changes for logging
        old_bal_agent = self.agent.balance
        old_bal_opp = self.opponent.balance
        
        roll_number, was_double = player.roll()
        self.game.active_player.earn(roll_number)
        self.game.inactive_player.earn(roll_number)
        
        if self.verbose:
            print(f"\n🎲 {player_name} rolls a {roll_number} {'(Double!)' if was_double else ''}")
            diff_agent = self.agent.balance - old_bal_agent
            diff_opp = self.opponent.balance - old_bal_opp
            
            if diff_agent > 0: print(f"   -> AI receives {diff_agent} coins.")
            if diff_opp > 0:   print(f"   -> YOU receive {diff_opp} coins.")

    def _get_obs(self):
        return self._build_obs_array(self.agent, self.opponent)

    def _get_obs_flipped(self):
        return self._build_obs_array(self.opponent, self.agent)

    def _build_obs_array(self, p1, p2):
        obs = []
        obs.append(p1.balance)
        obs.append(p2.balance)
        
        def count_cards(card_list):
            c = Counter([type(card) for card in card_list])
            return [c[t] for t in self.card_types]
            
        def count_market(establishment_counter):
            return [establishment_counter[t] for t in self.card_types]

        obs.extend(count_cards(p1.hand))
        obs.extend(count_cards(p2.hand))
        obs.extend(count_market(self.game.establishments))
        
        return np.array(obs, dtype=np.float32)