"""
ai_players.py
"""
import random
from machi_koro import Player

class RandomPlayer(Player):
    """
    A player that makes completely random choices.
    Used as an opponent for the AI during training.
    """
    def construct(self, available):
        # Select a random affordable card
        affordable = [c for c in available if self.has_funds_for(c(self, self.game))]
        if affordable:
            return random.choice(affordable)
        return None

    def choose_favorite_card(self, cards):
        # Required for Business Center (trade)
        if not cards:
            return None
        return random.choice(list(cards))

    def choose_least_favorite_card(self, cards):
        # Required for Business Center (trade)
        if not cards:
            return None
        return random.choice(list(cards))

class GymPlayer(Player):
    """
    The Reinforcement Learning Agent.
    """
    def construct(self, available):
        # Logic is handled by the Gym environment's 'step' function.
        return None

    def choose_favorite_card(self, cards):
        # Needed for Business Center trade
        if not cards:
            return None
        # Choose the most expensive card to receive
        return max(cards, key=lambda c: c.cost)

    def choose_least_favorite_card(self, cards):
        # Needed for Business Center trade
        if not cards:
            return None
        # Give away the cheapest card
        return min(cards, key=lambda c: c.cost)

class HumanPlayer(Player):
    """
    A human player that requests actions via the terminal.
    """
    def construct(self, available):
        print("\n--- YOUR TURN ---")
        print(f"You have {self.balance} coins.")
        print("Available cards:")
        
        # Filter affordable cards.
        affordable = [c for c in available if self.has_funds_for(c(self, self.game))]
        affordable = sorted(affordable, key=lambda c: c(self, self.game).cost)
        
        if not affordable:
            print("No affordable cards.")
            return None

        for i, card_class in enumerate(affordable):
            dummy = card_class(self, self.game)
            print(f"{i + 1}: {dummy.color.name} - {card_class.__name__} (Cost: {dummy.cost}, Icon: {dummy.symbol.name})")
        
        print("0: Pass (Do not buy)")

        while True:
            try:
                keuze = input("Choose a number: ")
                idx = int(keuze)
                if idx == 0:
                    return None
                if 1 <= idx <= len(affordable):
                    return affordable[idx - 1]
                print("Invalid number.")
            except ValueError:
                print("Please enter a number.")

    def choose_favorite_card(self, cards):
        # Business Center trade - which card to receive?
        if not cards: return None
        print("\nChoose a card to RECEIVE:")
        card_list = list(cards)
        for i, c in enumerate(card_list):
            print(f"{i}: {c.__class__.__name__} (Cost: {c.cost})")
        while True:
            try:
                idx = int(input("Choice number: "))
                if 0 <= idx < len(card_list):
                    return card_list[idx]
            except ValueError: pass

    def choose_least_favorite_card(self, cards):
        # Business Center trade - which card to give away?
        if not cards: return None
        print("\nChoose a card to GIVE AWAY:")
        card_list = list(cards)
        for i, c in enumerate(card_list):
            print(f"{i}: {c.__class__.__name__} (Cost: {c.cost})")
        while True:
            try:
                idx = int(input("Choice number: "))
                if 0 <= idx < len(card_list):
                    return card_list[idx]
            except ValueError: pass