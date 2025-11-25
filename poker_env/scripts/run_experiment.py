# poker_env/scripts/run_experiment.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import os
import random
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from poker.game import PokerGame
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def run_counterfactual_chain(model, tokenizer, base_dir, num_players,
                             starting_chips, dialogue_turns, counterfactual_seeds, stages):
    """
    For each directory, find the earliest stage snapshot and create counterfactual branches
    ONLY from that stage forward. Then recurse into each new cf directory.
    """

    # Determine earliest existing stage
    for idx, stage in enumerate(stages):
        if os.path.exists(os.path.join(base_dir, f"{stage}.json")):
            earliest_stage = stage
            earliest_index = idx
            break
    else:
        return  # no snapshots present

    # If earliest is RIVER, no more counterfactuals possible
    if earliest_stage == "RIVER":
        return

    next_stage = stages[earliest_index + 1]

    print(f"\n[CHAIN] Base: {base_dir} | Earliest stage: {earliest_stage} → next: {next_stage}")

    # Run CF seeds
    for cf_seed in range(counterfactual_seeds):
        cf_dir = os.path.join(base_dir, str(cf_seed))

        if os.path.exists(cf_dir):
            print(f"[SKIP] Exists: {cf_dir}")
            continue

        os.makedirs(cf_dir, exist_ok=True)
        print(f"[COUNTERFACTUAL] Creating: {cf_dir}")

        snapshot_path = os.path.join(base_dir, f"{earliest_stage}.json")

        cf_game = PokerGame(
            model=model,
            tokenizer=tokenizer,
            num_players=num_players,
            starting_chips=starting_chips,
            seed=cf_seed,
            snapshots_dir=cf_dir
        )

        cf_game.restore_from_snapshot(snapshot_path)
        cf_game.run_hand_from_stage(
            dialogue_turns=dialogue_turns,
            start_stage=next_stage
        )

        # Recurse into new CF branch
        run_counterfactual_chain(
            model=model,
            tokenizer=tokenizer,
            base_dir=cf_dir,
            num_players=num_players,
            starting_chips=starting_chips,
            dialogue_turns=dialogue_turns,
            counterfactual_seeds=counterfactual_seeds,
            stages=stages
        )

        

def main():
    parser = argparse.ArgumentParser(description="Run poker experiments and counterfactuals")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--starting_chips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--counterfactual_seeds", type=int, default=3)
    parser.add_argument("--snapshots_dir", type=str, default="/playpen-ssd/smerrill/deception/poker_env/poker/snapshots")
    parser.add_argument("--dialogue_turns", type=int, default=1)
    args = parser.parse_args()

    # load model/tokenizer
    max_seq_length = 5000
    device_map = {'': 0}
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        max_seq_length=max_seq_length,
        device_map=device_map,
        load_in_4bit=True,
        fix_tokenizer=True,
        offload_folder="./offload",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    main_dir = os.path.join(args.snapshots_dir, str(args.seed))
    # --- RUN MAIN GAME ---
    if os.path.exists(main_dir):
        print(f"Snapshots dir {main_dir} already exists. Skipping.")
    else:
        game = PokerGame(model, tokenizer, num_players=args.num_players, starting_chips=args.starting_chips,
                        seed=args.seed, snapshots_dir=os.path.join(args.snapshots_dir, str(args.seed)))
        print(f"=== Running main game with seed {args.seed} ===")
        game.run_hand_from_stage(dialogue_turns=args.dialogue_turns)

    # ——— FULL RECURSIVE COUNTERFACTUAL CHAIN ———
    print("\n=== Running FULL COUNTERFACTUAL MARKOV CHAIN ===")

    stages = ["PREFLOP", "FLOP", "TURN", "RIVER"]

    run_counterfactual_chain(
        model=model,
        tokenizer=tokenizer,
        base_dir=main_dir,
        num_players=args.num_players,
        starting_chips=args.starting_chips,
        dialogue_turns=args.dialogue_turns,
        counterfactual_seeds=args.counterfactual_seeds,
        stages=stages
    )

if __name__ == "__main__":
    main()
