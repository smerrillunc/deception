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
        self.models_dict = {}
        self.tokenizers_dict = {}

        ensure_dir(log_root)

    def _load_models(self, model_names):
        for model_name in model_names:
            from utils import load_model_and_tokenizer
            if model_name in self.models_dict.keys():
                print(f"Model {model_name} already loaded, skipping.")
                continue
            else:
                print(f"Loading model {model_name}...")
                model, tokenizer = load_model_and_tokenizer(model_name)
                self.models_dict[model_name] = model
                self.tokenizers_dict[model_name] = tokenizer
        return
    
    def run_single_game(self, model_names, cots,seed=0, n_cards=5):
        set_global_seed(seed)

        deck = self.deck_class(seed)

        self._load_models(model_names)
        # Do not write per-agent logs during runs; keep log_dir None
        agents = [self.agent_class(name=f"{c}",
                       cot=cots[c],
                       model_name=model_names[c],
                       model=self.models_dict[model_names[c]], 
                       tokenizer=self.tokenizers_dict[model_names[c]], 
                       seed=seed, 
                       log_dir=None) 
              for c in range(self.num_players)]

        # Do not create environment-level logs by default; snapshots will be saved directly under game folder
        env = BSEnvironment(agents, deck, seed=seed, log_dir=None)
        env.deal(n_cards=n_cards)

        game_dir = Path(self.log_root) / f"game_seed_{seed}"
        game_dir.mkdir(parents=True, exist_ok=True)

        snapshots = []
        # If snapshots already exist for this seed, skip running and report
        existing = list(game_dir.glob('turn_*.json'))
        if existing:
            print(f"Skipping run_single_game for seed {seed}: snapshots already exist in {game_dir}")
            return None, []
        while not env.game_over() and env.turn < self.max_steps:
            summary_play, summary_challenge = env.step()
            snapshot = env.get_snapshot()
            snapshots.append(snapshot)
            # save snapshot directly in the game folder (no snapshots subfolder)
            with open(game_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)
        return env, snapshots

    def run_monte_carlo(self, snapshot, num_sims=100, steps_per_sim=10):
        base_seed = snapshot.get("seed", None)
        # Place Monte Carlo results inside the parent seed folder. Each sim will create its own seed folder and
        # a Turn_{N} subfolder for the starting turn (no 'monte_carlo_turn_' prefix).
        monte_base = Path(self.log_root) / f"game_seed_{base_seed}"
        monte_base.mkdir(parents=True, exist_ok=True)

        model_names = snapshot.get("model_names", None)
        cots = snapshot.get("cots", None)

        self._load_models(model_names)

        last_env = None
        last_snapshots = []
        for sim_idx in tqdm.tqdm(range(num_sims)):
            print("Starting MC sim: ", sim_idx)
            seed = base_seed + sim_idx
            random.seed(seed)

            # Do not write per-agent logs during MC sims
            agents = [self.agent_class(name=f"{c}",
                                      model_name=model_names[c],
                                      cot=cots[c],
                                      model=self.models_dict[model_names[c]],
                                      tokenizer=self.tokenizers_dict[model_names[c]],
                                      seed=seed,
                                      log_dir=None)
                      for c in range(self.num_players)]

            # Deck not used yet
            deck = copy.deepcopy(snapshot.get("deck_state", None))
            # Create a per-sim output folder: game_seed_{base_seed}/seed_{seed}/Turn_{start_turn}
            sim_dir = monte_base / f"Turn_{snapshot['turn']}" / f"seed_{seed}"
            # If the sim output folder already has snapshots, skip this sim
            if sim_dir.exists() and any(sim_dir.glob('turn_*.json')):
                print(f"Skipping MC sim {sim_idx} (seed {seed}): snapshots already exist in {sim_dir}")
                continue
            sim_dir.mkdir(parents=True, exist_ok=True)
            env = BSEnvironment.from_snapshot(snapshot, agents, deck, log_dir=None)
            env.seed = seed
            snapshots = []

            total_steps = 0
            while not env.game_over() and env.turn < self.max_steps and total_steps < steps_per_sim:
                summary_play, summary_challenge = env.step()
                snap = env.get_snapshot()
                snapshots.append(snap)
                # save snapshot into the per-sim folder
                with open(sim_dir / f"turn_{env.turn-1}.json", "w") as f:
                    json.dump(snap, f, indent=2)
                total_steps += 1

            last_env = env
            last_snapshots = snapshots

        return last_env, last_snapshots

    def run_seed_trajectory(self, model_names, cots, seeds, n_cards=5, output_dir_name="seed_trajectory"):
        """
        Run a single game where each environment step uses a different random seed.
        seeds: list of integers; seeds[i] is used before step i.
        """
        assert len(seeds) > 0, "Must provide at least one seed."

        # --- Step 1: Initialize with seeds[0] ---
        init_seed = seeds[0]
        set_global_seed(init_seed)

        deck = self.deck_class(init_seed)

        # Load models
        self._load_models(model_names)

        # Create agents
        agents = [
            self.agent_class(
                name=f"{i}",
                cot=cots[i],
                model_name=model_names[i],
                model=self.models_dict[model_names[i]],
                tokenizer=self.tokenizers_dict[model_names[i]],
                seed=init_seed,   # initial seed
                log_dir=None
            )
            for i in range(self.num_players)
        ]

        # Initialize environment
        env_dir = Path(self.log_root) / output_dir_name
        env_dir.mkdir(parents=True, exist_ok=True)

        # If snapshots already exist for this seed_trajectory folder, skip the run
        if any(env_dir.glob('turn_*.json')):
            print(f"Skipping seed_trajectory run: snapshots already exist in {env_dir}")
            return None, []

        env = BSEnvironment(agents, deck, seed=init_seed, log_dir=None)
        env.deal(n_cards=n_cards)

        snapshots = []

        # --- Step 2: Run one step per seed ---
        for step_idx, seed in enumerate(seeds):

            # If game ended early, stop
            if env.game_over() or env.turn >= self.max_steps:
                break

            # Step 3: update all agent seeds
            for agent in env.agents:
                agent.seed = seed
            env.seed = seed   # optional, if env also uses randomness

            set_global_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # --- Step 4: take a step ---
            summary_play, summary_challenge = env.step()

            snapshot = env.get_snapshot()
            snapshots.append(snapshot)

            # Save snapshot directly in the trajectory folder
            with open(env_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)

        return env, snapshots