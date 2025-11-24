import os
import json
from copy import deepcopy
import random
import sys
sys.path.append("/playpen-ssd/smerrill/deception/BS/src")
from utils.io import ensure_dir
from simulation.bs_environment import BSEnvironment
from pathlib import Path
import copy
from game.deck import Deck
from treys import Card
import numpy as np
import torch

class GameRunner:
    def __init__(self, deck_class, agent_class, num_players=2, log_root="games"):
        self.deck_class = deck_class
        self.agent_class = agent_class
        self.num_players = num_players
        self.log_root = log_root
        ensure_dir(log_root)

    def run_single_game(self, model, tokenizer, seed=0, n_cards=5, debug_config=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        deck = self.deck_class()
        agents = [self.agent_class(name=f"{c}", model=model, tokenizer=tokenizer, seed=seed, debug_config=debug_config,
                                   log_dir=os.path.join(self.log_root, f"game_seed_{seed}", "logs")) 
                  for c in "AB"[:self.num_players]]
        env = BSEnvironment(agents, deck, debug_config=debug_config,
                            log_dir=os.path.join(self.log_root, f"game_seed_{seed}"))
        env.deal(n_cards=n_cards)

        game_dir = Path(self.log_root) / f"game_seed_{seed}" / "snapshots"
        game_dir.mkdir(parents=True, exist_ok=True)

        snapshots = []
        while not env.game_over():
            summary_play, summary_challenge = env.step()
            snapshot = env.get_snapshot()
            snapshots.append(snapshot)
            # save snapshot
            with open(game_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)
        return env, snapshots

    def run_monte_carlo(self, snapshot, base_seed=0, num_sims=100, n_cards=5):
        monte_dir = Path(self.log_root) / f"game_seed_{base_seed}" / f"monte_carlo_turn_{snapshot['turn']}"
        monte_dir.mkdir(parents=True, exist_ok=True)

        sim_files = []
        for sim_idx in range(num_sims):
            seed = base_seed + sim_idx
            random.seed(seed)
            # recreate agents and deck
            agents = [self.agent_class(name=a_name, model=None, tokenizer=None, seed=seed,
                                       log_dir=monte_dir) for a_name in snapshot["hands"].keys()]
            deck = copy.deepcopy(snapshot.get("deck_state", None))  # implement if needed
            env = BSEnvironment.from_snapshot(snapshot, agents, deck, log_dir=monte_dir)
            sim_results = []

            while not env.game_over():
                summary_play, summary_challenge = env.step()
                sim_results.append({
                    "turn": env.turn,
                    "play": summary_play,
                    "challenge": summary_challenge,
                    "pile": [Card.int_to_str(c) for c in env.pile],
                    "hands": {a.name: [Card.int_to_str(c) for c in a.hand] for a in env.agents}
                })
            sim_file = monte_dir / f"mc_seed_{seed}.json"
            with open(sim_file, "w") as f:
                json.dump(sim_results, f, indent=2)
            sim_files.append(sim_file)
        return sim_files
