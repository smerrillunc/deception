# poker_sim/game.py
from typing import List, Tuple, Dict, Any
from .deck import Deck
from .llm_agent import LLMAgent
from treys import Evaluator, Card
from copy import deepcopy
import random
import numpy as np
import torch
from .utils import ensure_dir
import json
import os

class PokerGame:
    def __init__(self, model, tokenizer, num_players:int=2, starting_chips:int=100, seed:int=42, snapshots_dir="snapshots"):
        self.model = model
        self.tokenizer = tokenizer
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.seed = seed
        self.snapshots_dir = snapshots_dir
        self.evaluator = Evaluator()
        self.reset()

    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.deck = Deck(seed=self.seed)
        self.agents = [LLMAgent(f"Player{i+1}", model=self.model, tokenizer=self.tokenizer, seed=self.seed, chips=self.starting_chips) for i in range(self.num_players)]
        for a in self.agents:
            a.hand = [self.deck.draw(1), self.deck.draw(1)]
        self.board = [self.deck.draw(1) for _ in range(5)]
        self.pot = 0
        self.current_bet = 0
        self.dialogue_history = []

    def format_conversation_history(self, current_stage: str) -> str:
        # simple formatter (similar to original)
        output = []
        stage_buffer = []
        last_stage = None
        for entry in self.dialogue_history:
            stg = entry.get("stage", "")
            agent = entry.get("agent", "Unknown")
            utter = entry.get("utterance") or entry.get("action")
            action = entry.get("action")
            if action == "dialogue" or entry.get("action")=="dialogue":
                if stg != last_stage:
                    if stage_buffer:
                        output.append("\n".join(stage_buffer))
                        stage_buffer = []
                    stage_buffer.append(f"=== {stg} Dialogue ===")
                    last_stage = stg
                stage_buffer.append(f"{agent}: {entry.get('utterance','')}")
        if stage_buffer:
            output.append("\n".join(stage_buffer))
        return "\n\n".join(output).strip()

    def betting_round(self, stage: str, max_new_tokens:int=256):
        num_players = len(self.agents)
        committed = [0]*num_players
        round_over = False
        while not round_over:
            round_over = True
            for idx, agent in enumerate(self.agents):
                if agent.folded:
                    continue
                to_call = self.current_bet - committed[idx]
                other_chips = {a.name:a.chips for j,a in enumerate(self.agents) if j!=idx}
                history_text = self.format_conversation_history(stage)
                action_result = agent.decide_action(stage, self.visible_board(stage), other_chips, self.pot, to_call, history_text, max_new_tokens, self.seed)
                if isinstance(action_result, tuple):
                    action_data, reasoning = action_result
                else:
                    action_data = action_result
                    reasoning = action_data.get("reasoning", "")
                act = action_data.get("action", "check").lower()
                amt = min(int(action_data.get("amount", 0)), agent.chips)
                # legality
                if to_call > 0:
                    if act=="check":
                        act="call"; amt=to_call
                    elif act=="call":
                        amt=to_call
                    elif act=="raise":
                        amt=max(amt, to_call+1)
                else:
                    if act not in ["check","raise"]:
                        act="check"; amt=0
                # execute
                if act=="fold":
                    agent.folded=True
                elif act=="call":
                    agent.chips-=amt
                    committed[idx]+=amt
                    self.pot+=amt
                elif act=="raise":
                    agent.chips-=amt
                    committed[idx]+=amt
                    self.pot+=amt
                    self.current_bet = committed[idx]
                    round_over=False
                elif act=="check":
                    pass
                self.dialogue_history.append({
                    "agent": agent.name,
                    "stage": stage,
                    "action": act,
                    "amount": amt,
                    "reasoning": reasoning
                })
                print(f"{agent.name}: {act} {amt}, chips={agent.chips}")
            active_committed = [committed[i] for i,a in enumerate(self.agents) if not a.folded]
            if len(active_committed)>0 and len(set(active_committed))==1:
                round_over=True
        return self.pot, committed

    def visible_board(self, stage):
        if stage == "PREFLOP":
            return []
        elif stage == "FLOP":
            return self.board[:3]
        elif stage == "TURN":
            return self.board[:4]
        elif stage == "RIVER":
            return self.board[:5]
        return []

    def run_hand_from_stage(self, start_stage="PREFLOP", dialogue_turns=1, verbose=True):
        stages = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        start_idx = stages.index(start_stage)
        for stage_name in stages[start_idx:]:
            # update win% after flop
            if stage_name != "PREFLOP":
                visible_board = self.visible_board(stage_name)
                for a in self.agents:
                    rank = self.evaluator.evaluate(a.hand, visible_board)
                    rank_class = self.evaluator.get_rank_class(rank)
                    hand_str = self.evaluator.class_to_string(rank_class)
                    pct = 1.0 - self.evaluator.get_five_card_rank_percentage(rank)
                    a.win_pct = f"{pct*100:.2f}%"
                    if verbose:
                        print(f"{a.name} win%={a.win_pct} hand={hand_str}")

            # dialogue
            if dialogue_turns > 0:
                for r in range(dialogue_turns):
                    for speaker_idx, speaker in enumerate(self.agents):
                        history_text = self.format_conversation_history(stage_name)
                        utter, reasoning = speaker.generate_dialogue(stage_name, self.visible_board(stage_name), 
                                                                    {a.name:a.chips for i,a in enumerate(self.agents) if i!=speaker_idx},
                                                                    self.pot, history_text, seed=self.seed)
                        self.dialogue_history.append({
                            "agent": speaker.name,
                            "stage": stage_name,
                            "action": "dialogue",
                            "utterance": utter,
                            "reasoning": reasoning
                        })
                        print(f"{speaker.name}: {utter}\n{'-'*80}")

            # betting
            self.betting_round(stage_name, max_new_tokens=256)
            # one active left -> win
            active_agents = [a for a in self.agents if not a.folded]
            if len(active_agents) == 1:
                winner = active_agents[0]
                winner.chips += self.pot
                print(f"\n*** {winner.name} wins {self.pot} chips by opponent folding! ***")
                self.snapshot(stage_name)
                return
            if verbose:
                print(f"\nPot now: {self.pot}")
                for a in self.agents:
                    print(f"{a.name}: chips={a.chips}, folded={a.folded}")
            self.snapshot(stage_name)
        # showdown
        active_agents = [a for a in self.agents if not a.folded]
        best_rank = None
        winner = None
        for a in active_agents:
            rank = self.evaluator.evaluate(a.hand, self.board)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                winner = a
        if winner:
            winner.chips += self.pot
            print(f"\n*** {winner.name} wins {self.pot} chips at showdown! ***")
        self.snapshot("SHOWDOWN")

    def snapshot(self, stage: str):
        ensure_dir(self.snapshots_dir)
        snap = {
            "seed": self.seed,
            "stage": stage,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "board": deepcopy(self.board),
            "deck": deepcopy(self.deck.cards),
            "dialogue_history": deepcopy(self.dialogue_history),
            "agents": [
                {
                    "name": a.name,
                    "chips": a.chips,
                    "folded": a.folded,
                    "hand": deepcopy(a.hand),
                    "last_action": deepcopy(a.last_action),
                    "win_pct": a.win_pct
                } for a in self.agents
            ]
        }
        path = PokerGame.save_snapshot(self.snapshots_dir, stage, snap)
        print("✅ Saved snapshot to", path)
        return path

    def restore_from_snapshot(self, snapshot_path: str):
        snap = PokerGame.load_snapshot(snapshot_path)
        self.seed = snap.get("seed", self.seed)
        self.pot = snap.get("pot", 0)
        self.current_bet = snap.get("current_bet", 0)
        self.board = deepcopy(snap.get("board", []))
        deck_cards = deepcopy(snap.get("deck", []))
        self.dialogue_history = deepcopy(snap.get("dialogue_history", []))
        # rebuild deck object
        self.deck = Deck(seed=self.seed)
        self.deck.cards = deck_cards
        self.agents = []
        for a in snap.get("agents", []):
            agent = LLMAgent(a["name"], model=self.model, tokenizer=self.tokenizer, seed=self.seed, chips=a["chips"])
            agent.hand = a.get("hand", [])
            agent.folded = a.get("folded", False)
            agent.last_action = a.get("last_action", None)
            agent.win_pct = a.get("win_pct", '')
            self.agents.append(agent)
        return self
    
    @staticmethod
    def save_snapshot(snapshots_dir: str, stage: str, snap: Dict[str, Any]) -> str:
        os.makedirs(snapshots_dir, exist_ok=True)
        path = os.path.join(snapshots_dir, f"{stage}.json")
        with open(path, "w") as f:
            json.dump(snap, f, indent=2)
        return path

    @staticmethod
    def load_snapshot(snapshot_path: str) -> Dict[str, Any]:
        with open(snapshot_path, "r") as f:
            return json.load(f)

