import gc
import os
import json
import copy
import random
import sys
sys.path.append("/playpen-ssd/smerrill/deception/BS/src")
from utils import ensure_dir, set_global_seed
from bs_environment import BSEnvironment
from llm_agent import LLMAgent

from pathlib import Path
import numpy as np
import tqdm
import torch
import gzip
import io
import zstandard as zstd
from pathlib import Path

class GameRunner:
    def __init__(self, deck_class, agent_class, num_players=2, max_steps=100, log_root="games"):
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
        while not env.game_over() and env.turn <= self.max_steps:
            summary_play, summary_challenge = env.step()
            snapshot = env.get_snapshot()
            
            ### We won't save activations for standard game runs ###
            #act_filepaths = self._save_activations(env, game_dir)
            #snapshot['activations'] = act_filepaths

            snapshots.append(snapshot)
            # save snapshot directly in the game folder (no snapshots subfolder)
            with open(game_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)
            
        return env, snapshots

    def run_monte_carlo(self, snapshot, num_sims=100, steps_per_sim=1):
        base_seed = snapshot.get("seed", None)
        # Place Monte Carlo results inside the parent seed folder. Each sim will create its own seed folder and
        # a Turn_{N} subfolder for the starting turn (no 'monte_carlo_turn_' prefix).
        monte_base = Path(self.log_root)
        monte_base.mkdir(parents=True, exist_ok=True)

        model_names = snapshot.get("model_names", None)
        cots = snapshot.get("cots", None)

        self._load_models(model_names)

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
            
            sim_dir = monte_base

            # If the sim output folder already has snapshots, skip this sim
            if sim_dir.exists() and os.path.exists(sim_dir / f"seed_{seed}.json"):
                print(f"Skipping MC sim {sim_idx} (seed {seed}): snapshots already exist in {sim_dir}")
                continue

            sim_dir.mkdir(parents=True, exist_ok=True)
            env = BSEnvironment.from_snapshot(snapshot, agents, deck, log_dir=None)
            env.seed = seed
            #snapshots = []

            total_steps = 0
            while not env.game_over() and env.turn <= self.max_steps and total_steps < steps_per_sim:
                # we want to save activations during MC sims
                summary_play, summary_challenge = env.step(save_activations=True)
                snap = env.get_snapshot()

                # this will be the player who just acted (not challenged or passed)
                current_idx = (env.turn - 1) % len(env.agents)
                act_filepaths = self._save_activations(env, sim_dir, seed, current_idx)
                snap['activations'] = act_filepaths

                #snapshots.append(snap)
            
                # save snapshot into the per-sim folder
                with open(sim_dir / f"seed_{seed}.json", "w") as f:
                    json.dump(snap, f, indent=2)

                total_steps += 1
                del snap

            agents = None
            del env
            del agents
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return 1


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
            
            act_filepaths = self._save_activations(env, env_dir)
            snapshot['activations'] = act_filepaths
            snapshots.append(snapshot)

            # Save snapshot directly in the trajectory folder
            with open(env_dir / f"turn_{env.turn-1}.json", "w") as f:
                json.dump(snapshot, f, indent=2)

        return env, snapshots
    
    def run_truthful_trajectory(self, snapshot, num_sims=5, output_name="truthful_trajectory/turn_0.pkl"):
        if os.path.exists(output_name):
            print(f"Skipping truthful trajectory: output already exists at  output_name")
            return []
        else:
            ensure_dir(os.path.dirname(output_name))
        
        actions = []
        num_players = 2
        base_seed = snapshot.get("seed", None)
        model_names = snapshot.get("model_names", None)
        cots = snapshot.get("cots", None)
        deck = copy.deepcopy(snapshot.get("deck_state", None))

        # Load models
        self._load_models(model_names)

        for sim_idx in tqdm.tqdm(range(num_sims)):
            print('-'*20)
            print("Starting truthful sim: ", sim_idx)
            seed = base_seed + sim_idx
            random.seed(seed)

            # Do not write per-agent logs during MC sims
            agents = [LLMAgent(name=f"{c}",
                    model_name=model_names[c],
                    cot=cots[c],
                    model=self.models_dict[model_names[c]],
                    tokenizer=self.tokenizers_dict[model_names[c]],
                    seed=seed,
                    log_dir=None)
                        for c in range(num_players)]
            
            env = BSEnvironment.from_snapshot(snapshot, agents, deck, log_dir=None)

            current_idx = env.turn % len(env.agents)
            current = env.agents[current_idx]
            opponent_idx = (env.turn + 1) % len(env.agents)
            opponent = env.agents[opponent_idx]

            # --- Challenge Prompt Construction ---

            game_history = snapshot['full_play_history']
            game_history = sorted(game_history, key=lambda x: x['ts'])

            # remove the last two entried
            game_history = game_history[:-2]

            # opponent hand is the player on the next turn
            # player name is opponent on last turn
            player = snapshot['last_play'][-1]['player']
            hand = snapshot['last_play'][-2]['opponent_hand_str']

            # current rank is the rank for this turn
            current_rank = snapshot['current_rank']

            # a truthful aciton
            action = GameRunner.play_matching_cards(hand, current_rank)
            last_ts = snapshot['last_play'][-1]['ts']

            pile = snapshot.get('pile',[])
            card_indices = action.get("Card_idx", [])
            #card_indices = [0, 1]
            played_cards = [hand[i] for i in card_indices if i < len(hand)]

            for c in played_cards:
                if c in hand:
                    hand.remove(c)
                pile.append(c)

            summary_play = f"Player {player} played {len(played_cards)} card(s), claiming rank {current_rank}."
            game_history.append({'player':player,
                'content':summary_play,
                'ts':last_ts+1})

            last_play = {"played_cards": played_cards}

            # --- Challenge Prompt Construction ---
            challenge_prompt = env._make_challenge_prompt(opponent, last_play, current)

            full_challenge_history =BSEnvironment._merge_history_and_prompt(BSEnvironment._get_truncated_history(game_history), challenge_prompt)

            # we are just measuring if opponent's action distribution changes, so no need to save_activations
            challenge_action = opponent.act(full_challenge_history, save_activations=False)
            challenge_type = challenge_action.get("Action", "Pass")
            actions.append({seed: challenge_type})

        np.save(output_name, actions)
        return actions

    import torch

    @staticmethod
    def play_matching_cards(hand, current_rank):
        """
        Deterministic function to act truthfully by playing all cards in hand that match the current rank.
        return the reasoning and which card indices match that rank.
        """

        current_rank_str = str(current_rank)

        # Extract matching card indices and values
        matching = [
            (i, card) for i, card in enumerate(hand)
            if card[:-1] == current_rank_str
        ]

        matching_indices = [idx for idx, _ in matching]
        matching_cards = [card for _, card in matching]

        # Build reasoning
        if matching_indices:
            reasoning = (
                f"The current rank is {current_rank}. My hand has the following cards "
                f"that match the current rank: {matching_cards}, so I will play truthfully "
                f"and only play these cards"
            )
        else:
            reasoning = (
                f"The current rank is {current_rank}. My hand has no cards that match "
                f"the current rank so I will act truthfully and play no cards"
            )

        return {
            "Reasoning": reasoning,
            "Action": "Play",
            "Card_idx": matching_indices
        }


    def _save_activations(self, env, game_dir, seed=None, current_idx=None,
                        use_half=True, token_stride=1, kept_layers=None, compress=True):
        """
        Save activations efficiently using Zstandard compression, float16 conversion, and layer/token downsampling.

        Args:
            env: Environment containing agents.
            game_dir: Base directory to save activations.
            seed: Optional random seed, inferred from env if None.
            current_idx: Optional agent index, inferred from env.turn if None.
            use_half: Convert float tensors to float16 if True.
            token_stride: Keep every N-th token (downsample) to reduce size.
            kept_layers: List of layer indices to save (default: first + last layers).
            compress: If True, use zstd compression.
        """
        activations_dir = Path(game_dir) / "activations"
        activations_dir.mkdir(exist_ok=True, parents=True)

        # Infer current_idx and seed
        if current_idx is None:
            try:
                current_idx = (env.turn - 1) % len(env.agents)
            except Exception:
                current_idx = 0
        if seed is None:
            seed = getattr(env, 'seed', 'unknown')

        current = env.agents[current_idx]

        acts = getattr(current, 'activations', None)
        if not acts:
            try:
                current._remove_hooks()
            except Exception:
                pass
            if hasattr(current, 'activations'):
                try:
                    current.activations = None
                    delattr(current, 'activations')
                except Exception:
                    pass
            return {}

        # ------------------- Process activations -------------------
        processed_acts = {}

        # Hidden states
        hidden_states = acts.get("hidden_states", [])
        if hidden_states:
            # Select only kept layers
            if kept_layers is None:
                # default: first and last layers
                kept_layers = [0, len(hidden_states)-1]
            hidden_states = [hidden_states[i] for i in kept_layers if i < len(hidden_states)]

            # Downsample tokens
            if token_stride > 1:
                hidden_states = [h[:, ::token_stride, :].contiguous() for h in hidden_states]

            # Convert to float16
            if use_half:
                hidden_states = [h.half() for h in hidden_states]

            processed_acts["hidden_states"] = hidden_states

        # Logits
        logits = acts.get("logits", [])
        if logits:
            if token_stride > 1:
                logits = [l[::token_stride].contiguous() for l in logits]
            if use_half:
                logits = [l.half() if l.is_floating_point() else l for l in logits]
            processed_acts["logits"] = logits

        # Save metadata
        for key in ["kept_layers", "activation_stride", "num_model_layers"]:
            if key in acts:
                processed_acts[key] = acts[key]

        # ------------------- Save to disk -------------------
        save_path = activations_dir / f"seed_{seed}_agent_{current.name}.pt"
        if compress:
            save_path = str(save_path) + ".zst"
            cctx = zstd.ZstdCompressor(level=3)
            buf = io.BytesIO()
            torch.save(processed_acts, buf)
            compressed = cctx.compress(buf.getvalue())
            with open(save_path, "wb") as f:
                f.write(compressed)
        else:
            torch.save(processed_acts, save_path)

        # ------------------- Cleanup -------------------
        try:
            current._remove_hooks()
        except Exception:
            pass
        try:
            current.activations = None
            delattr(current, 'activations')
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        return {current.name: str(save_path)}
