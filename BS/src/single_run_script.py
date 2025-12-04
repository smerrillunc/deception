#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single BS game with LLM agents."
    )

    parser.add_argument(
        "--src-path",
        type=str,
        default="/playpen-ssd/smerrill/deception/BS/src",
        help="Path to add to PYTHONPATH so local modules (Deck, LLMAgent, etc.) can import.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        help="Base model name for players.",
    )

    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players in the game.",
    )

    parser.add_argument(
        "--cot",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether CoT is enabled for all players.",
    )

    parser.add_argument(
        "--log-root",
        type=str,
        default="games",
        help="Directory where game logs will be saved.",
    )

    parser.add_argument(
        "--n-cards",
        type=int,
        default=5,
        help="Number of cards dealt to each player initially.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps in the game.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debugging information.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    src_path = Path(args.src_path).resolve()
    sys.path.append(str(src_path))
    if args.verbose:
        print(f"✓ Added {src_path} to PYTHONPATH.")

    from deck import Deck
    from llm_agent import LLMAgent
    from bs_environment import BSEnvironment
    from runner import GameRunner

    if args.verbose:
        print("✓ Imported modules successfully.")

    base_model = args.base_model
    num_players = args.num_players

    model_names = [base_model] * num_players
    cot_flag = args.cot.lower() == "true"
    cots = [cot_flag] * num_players

    runner = GameRunner(
        deck_class=Deck,
        agent_class=LLMAgent,
        log_root=args.log_root,
        max_steps=args.max_steps,

    )

    print(f"Running game with seed {args.seed:04d}")
    env, snapshots = runner.run_single_game(
        model_names,
        cots,
        seed=args.seed,
        n_cards=args.n_cards,
    )

    if args.verbose:
        print("✓ Game completed.")
        print("Environment:", env)
        print("Snapshots:", len(snapshots))

    return env, snapshots


if __name__ == "__main__":
    main()
