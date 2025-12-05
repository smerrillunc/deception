#!/usr/bin/env python3

import os
import sys
import json
import random
import argparse
from pathlib import Path
import tqdm

# ---- Project Imports ----
sys.path.append("/playpen-ssd/smerrill/deception/BS/src")

from deck import Deck
from llm_agent import LLMAgent
from runner import GameRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo rollouts over saved BS game snapshots"
    )

    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to directory containing game_seed folders with JSON snapshots"
    )

    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Root directory where MC outputs will be saved"
    )

    parser.add_argument(
        "--num_sims",
        type=int,
        default=10,
        help="Number of Monte Carlo sims per snapshot"
    )

    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Optional cap on number of turns processed per seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result_path = Path(args.result_path)
    log_root = Path(args.log_root)

    assert result_path.exists(), f"Result path not found: {result_path}"

    game_seeds = [x for x in os.listdir(result_path) if not x.startswith(".")]
    random.shuffle(game_seeds)
    print(f"Found {len(game_seeds)} game seeds")

    runner = GameRunner(
        deck_class=Deck,
        agent_class=LLMAgent,
        log_root=log_root
    )

    for game_seed in game_seeds:
        path = os.path.join(result_path, game_seed)
        turns = os.listdir(path)
        turns = sorted([x for x in turns if x.endswith('.json')])

        for turn_file in turns:
            with open(os.path.join(path, turn_file), 'r') as f:
                snapshot = json.load(f)
            print(f"Turn file: {os.path.join(path, turn_file)}")
            output_name = os.path.join(path, 'truthful', turn_file).replace('.json', '.npy')
            runner.run_truthful_trajectory(snapshot, num_sims=args.num_sims, output_name=output_name)
    print("\nTruthful batch completed successfully.")


if __name__ == "__main__":
    main()
