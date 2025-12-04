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
        default=20,
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

    for game_seed in tqdm.tqdm(game_seeds, desc="Game Seeds"):
        seed_path = result_path / game_seed

        if not seed_path.is_dir():
            continue

        turn_files = sorted([
            x for x in os.listdir(seed_path)
            if x.endswith(".json")
        ])

        if args.max_turns is not None:
            turn_files = turn_files[:args.max_turns]

        for turn_file in turn_files:
            turn_path = seed_path / turn_file

            print(f"\n▶ Running MC for: {turn_path}")

            with open(turn_path, "r") as f:
                snapshot = json.load(f)

            # Each snapshot gets its own MC output folder
            snapshot_log_root = turn_path.with_suffix("")
            snapshot_log_root.mkdir(parents=True, exist_ok=True)

            runner.log_root = str(snapshot_log_root)

            runner.run_monte_carlo(
                snapshot=snapshot,
                num_sims=args.num_sims
            )

    print("\n✅ Monte Carlo batch completed successfully.")


if __name__ == "__main__":
    main()
