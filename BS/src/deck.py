# poker_sim/deck.py
from treys import Card
import random
from typing import List

class Deck:
    _FULL_DECK: List[int] = []

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.shuffle()

    def shuffle(self):
        self.cards = Deck.get_full_deck()
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(self.cards)
        else:
            random.shuffle(self.cards)

    def draw(self, n=1):
        if n == 1:
            return self.cards.pop(0)
        return [self.draw() for _ in range(n)]

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        try:
            return Card.print_pretty_cards(self.cards)
        except Exception:
            return str(self.cards)

    @staticmethod
    def get_full_deck():
        if Deck._FULL_DECK:
            return list(Deck._FULL_DECK)

        # Only keep ranks 2–9; Note Treys uses 'T' for 10
        allowed_ranks = set("23456789")

        for rank in Card.STR_RANKS:
            if rank not in allowed_ranks:
                continue   # skip J, Q, K, A
            for suit in Card.CHAR_SUIT_TO_INT_SUIT.keys():
                Deck._FULL_DECK.append(Card.new(rank + suit))

        return list(Deck._FULL_DECK)
