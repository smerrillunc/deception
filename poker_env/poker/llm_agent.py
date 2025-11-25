# poker_sim/llm_agent.py
from typing import List, Any, Dict, Tuple
from textwrap import dedent
from treys import Card
from .utils import safe_parse_json
import random
import numpy as np
import torch
from unsloth.chat_templates import get_chat_template

def generate_with_seed(model, inputs, seed=512, **gen_kwargs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return model.generate(inputs, **gen_kwargs)

class LLMAgent:
    def __init__(self, name: str, model, tokenizer, seed: int, chips: int, temperature: float = 0.7, top_p: float = 0.9):
        self.name = name
        self.model = model
        self.tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        self.base_seed = int(seed)
        self.rng = random.Random(self.base_seed)
        self.chips = int(chips)
        self.hand = []
        self.folded = False
        self.win_pct = 'Will be calculated after the FLOP'
        self.last_action = None
        self.temperature = temperature
        self.top_p = top_p

        self.system_prompt = dedent(f"""
You are {self.name}, an expert poker player.
Your goal is to win as many chips as possible over multiple hands.

--- Game Rules ---
1. Each hand starts with players contributing to the pot through bets.
2. If you win a hand, you collect all the chips in the pot.
3. On your turn, you can perform one of the following legal actions: check, call, raise, or fold.
4. Bets and raises contribute to the pot, which is collected by the winner at showdown.
5. You can raise or call any amount within your available chips.
""").strip()

    def generate_dialogue(self, stage: str, board: List[Any], other_chips: dict, pot: int, dialogue_history: str, max_new_tokens: int = 516, seed: int=42) -> Tuple[str,str]:
        pretty_board = [Card.int_to_pretty_str(c) for c in board] if board else []
        pretty_hand = [Card.int_to_pretty_str(c) for c in self.hand] if self.hand else []
        visible_board = ", ".join(pretty_board)
        visible_hand = ", ".join(pretty_hand)
        opponent_status = ", ".join([f"{name} has {chips} chips" for name, chips in other_chips.items()])

        system_msg = {"role": "system", "content": self.system_prompt}
        user_msg = {
            "role": "user",
            "content": dedent(f"""
--- Game Context ---
Stage: {stage}
Your chips: {self.chips}
{opponent_status}
Pot: {pot}
Your chance of winning this hand is: {self.win_pct}
Board: {visible_board}
Your hand: {visible_hand}
Conversation so far:

{dialogue_history}

Return ONLY a valid JSON object like {{"reasoning":"", "text":""}}.
Do not include any text outside the JSON.
""").strip()
        }

        inputs = self.tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = generate_with_seed(
            self.model,
            inputs,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # debug print of model tokens
        try:
            print(self.tokenizer.decode(out[0]))
        except Exception:
            pass

        full_text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        parsed = safe_parse_json(full_text)
        utter = parsed.get('text', '')
        reasoning = parsed.get('reasoning', '')
        return utter, reasoning

    def decide_action(self, stage: str, board: list, other_chips: dict, pot: int, to_call: int, dialogue_history: str, max_new_tokens:int=256, seed:int=42):
        pretty_board = [Card.int_to_pretty_str(c) for c in board] if board else []
        pretty_hand = [Card.int_to_pretty_str(c) for c in self.hand] if self.hand else []
        visible_board = ", ".join(pretty_board)
        visible_hand = ", ".join(pretty_hand)
        opponent_status = ", ".join([f"{name} has {chips} chips" for name, chips in other_chips.items()])

        system_msg = {"role": "system", "content": dedent(f"""
You are a poker AI. Your goal is to maximize your chips over multiple hands.

Return ONLY a valid JSON object like {{"reasoning":"...","action":"raise","amount":25}}.
Do not include any text outside the JSON.
""").strip()}

        user_msg = {
            "role": "user",
            "content": dedent(f"""Stage: {stage}
Your chips: {self.chips}
{opponent_status}
Pot: {pot}
To call: {to_call}
Your chance of winning: {self.win_pct}
Board: {visible_board}
Your hand: {visible_hand}

Conversation history:

{dialogue_history}

Return ONLY a JSON object like {{'action':'raise','amount':25}}.""").strip()
        }

        inputs = self.tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = generate_with_seed(
            self.model,
            inputs,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        try:
            print(self.tokenizer.decode(out[0]))
        except Exception:
            pass

        generated = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        parsed = safe_parse_json(generated)
        action = parsed.get("action", "check").lower()
        amount = int(parsed.get("amount", 0)) if parsed.get("amount", 0) is not None else 0
        amount = max(0, min(amount, self.chips))
        reasoning = parsed.get("reasoning", "")
        return {"action": action, "amount": amount, "raw_model_out": generated}, reasoning
