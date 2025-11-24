import os, time, json, copy
from ..utils.io import write_json, append_jsonl, ensure_dir
from ..replay.snapshot import save_full_snapshot
from ..utils.random_tools import set_global_seed

class ReplayRunner:
    def __init__(self, snapshot_path, player_factories, out_dir='replays'):
        self.snapshot_path = snapshot_path
        self.player_factories = player_factories
        self.out_dir = out_dir
        ensure_dir(self.out_dir)
        self._load_snapshot()

    def _load_snapshot(self):
        with open(self.snapshot_path,'r') as f:
            bundle = json.load(f)
        self.base_state = bundle.get('state', {})
        self.meta = bundle.get('meta', {})

    def run_replay(self, seed, intervene_at=None, intervention_kind='force_truth', max_turns=200):
        set_global_seed(seed)
        agents = [fac(i) for i, fac in enumerate(self.player_factories)]
        # restore hands if available (strings of Card.int_to_str)
        for ai, a in enumerate(agents):
            h = self.base_state.get('agents_hands', {}).get(a.name, [])
            a.hand = list(h)
        replay_log = {'seed': seed, 'turns': []}
        for t in range(max_turns):
            obs = {'prompt': f"Replay turn {t} from snapshot, seed {seed}"}
            for a in agents:
                act = a.act(obs, t)
                replay_log['turns'].append({'turn': t, 'agent': a.name, 'action': act})
        fname = os.path.join(self.out_dir, f'replay_{seed}.json')
        write_json(replay_log, fname)
        return fname
