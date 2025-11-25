import os
import json
import copy
import random
import sys
sys.path.append("/playpen-ssd/smerrill/deception/BS/src")
from utils import ensure_dir, set_global_seed
from bs_environment import BSEnvironment

from pathlib import Path
import numpy as np
import tqdm

class GameRunner:
    def __init__(self, deck_class, agent_class, num_players=2, max_steps=20, log_root="games"):
        self.deck_class = deck_class
        self.agent_class = agent_class
        self.num_players = num_players
        self.log_root = log_root
        self.max_steps = max_steps
        ensure_dir(log_root)

    def _load_models(self, model_names):
        models_dict = {}
        tokenizers_dict = {}
        for model_name in model_names:
            from utils import load_model_and_tokenizer
            if model_name in models_dict.keys():
                print(f"Model {model_name} already loaded, skipping.")
                continue
            else:
                print(f"Loading model {model_name}...")
                model, tokenizer = load_model_and_tokenizer(model_name)
                models_dict[model_name] = model
                tokenizers_dict[model_name] = tokenizer

        return models_dict, tokenizers_dict
    
    def run_single_game(self, model_names, cots,seed=0, n_cards=5):
        set_global_seed(seed)

        deck = self.deck_class()

        models_dict = {}
        tokenizers_dict = {}
        for model_name in model_names:
            from utils import load_model_and_tokenizer
            if model_name in models_dict:
                continue

            model, tokenizer = load_model_and_tokenizer(model_name)
            models_dict[model_name] = model
            tokenizers_dict[model_name] = tokenizer

        models_dict, tokenizers_dict = self._load_models(model_names)
        agents = [self.agent_class(name=f"{c}",
                                   cot=cots[c],
                                   model_name=model_names[c],
                                   model=models_dict[model_names[c]], 
                                   tokenizer=tokenizers_dict[model_names[c]], 
                                   seed=seed, 
                                   log_dir=os.path.join(self.log_root, f"game_seed_{seed}", "logs")) 
                  for c in range(self.num_players)]

        env = BSEnvironment(agents, deck, seed=seed, log_dir=os.path.join(self.log_root, f"game_seed_{seed}"))
        env.deal(n_cards=n_cards)

        game_dir = Path(self.log_root) / f"game_seed_{seed}" / "snapshots"
        game_dir.mkdir(parents=True, exist_ok=True)

        snapshots = []
        while not env.game_over() and env.turn < self.max_steps:
            summary_play, summary_challenge = env.step()
            snapshot = env.get_snapshot()
            snapshots.append(snapshot)
            # save snapshot
            with open(game_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)
        return env, snapshots

    def run_monte_carlo(self, snapshot, num_sims=100, steps_per_sim=10):
        base_seed = snapshot.get("seed", None)
        monte_dir = Path(self.log_root) / f"game_seed_{base_seed}" / f"monte_carlo_turn_{snapshot['turn']}"
        monte_dir.mkdir(parents=True, exist_ok=True)

        model_names = snapshot.get("model_names", None)
        cots = snapshot.get("cots", None)

        models_dict, tokenizers_dict = self._load_models(model_names)

        sim_files = []
        for sim_idx in tqdm.tqdm(range(num_sims)):
            print("Starting MC sim: ", sim_idx)
            seed = base_seed + sim_idx
            random.seed(seed)
            # recreate agents and deck

            agents = [self.agent_class(name=f"{c}", 
                      model_name=model_names[c],
                      cot=cots[c],
                      model=models_dict[model_names[c]], 
                      tokenizer=tokenizers_dict[model_names[c]], 
                      seed=seed, 
                      log_dir=os.path.join(self.log_root, f"game_seed_{seed}", "logs")) 
                    for c in range(self.num_players)]

            # Deck not used yet
            deck = copy.deepcopy(snapshot.get("deck_state", None))  
            env = BSEnvironment.from_snapshot(snapshot, agents, deck, log_dir=monte_dir)
            env.seed = seed
            snapshots = []

            total_steps = 0
            while not env.game_over() and env.turn < self.max_steps and total_steps < steps_per_sim:
                summary_play, summary_challenge = env.step()
                snap = env.get_snapshot()
                snapshots.append(snap)
                # save snapshot
                os.makedirs(monte_dir / f"seed_{seed}", exist_ok=True)
                with open(monte_dir / f"seed_{seed}/turn_{env.turn-1}.json", "w") as f:
                    json.dump(snap, f, indent=2)
                total_steps += 1

        return env, snapshots
