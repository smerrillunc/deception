import os
import json
from pathlib import Path

import numpy as np
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results")

def list_game_seeds(result_path):
    p = Path(result_path)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith('.')])


def list_turns_for_seed(result_path, game_seed):
    seed_path = Path(result_path) / game_seed
    if not seed_path.exists():
        return []
    # Turns are typically directories (e.g., turn_0, turn_1)
    return sorted([x.name for x in seed_path.iterdir() if x.is_dir() and not x.name.startswith('.')])


def list_seed_entries_for_turn(result_path, game_seed, turn_name):
    turn_path = Path(result_path) / game_seed / turn_name
    if not turn_path.exists():
        return []
    # Prefer files ending with .json; if not present, list subentries
    json_files = sorted([x.name for x in turn_path.iterdir() if x.is_file() and x.suffix == '.json'])
    if json_files:
        # Return names without .json for consistency with notebook logic
        return [os.path.splitext(x)[0] for x in json_files]
    # Otherwise, list directories in the turn folder (each seed may be a folder)
    return sorted([x.name for x in turn_path.iterdir() if x.is_dir() and not x.name.startswith('.')])


def find_snapshot_file(result_path, game_seed, turn_name, seed_entry_name):
    base = Path(result_path) / game_seed / turn_name
    # If there is a file named seed_entry_name.json, prefer that
    candidate = base / (seed_entry_name + '.json')
    if candidate.exists():
        return str(candidate)

    # If seed_entry_name is a folder, look for json files inside
    folder = base / seed_entry_name
    if folder.exists() and folder.is_dir():
        # prefer any file named seed_entry_name.json inside folder
        candidate2 = folder / (seed_entry_name + '.json')
        if candidate2.exists():
            return str(candidate2)
        # otherwise pick first json file in folder
        jsons = sorted([x for x in folder.iterdir() if x.is_file() and x.suffix == '.json'])
        if jsons:
            return str(jsons[0])

    # As a last resort, pick the first json file inside the turn folder
    jsons = sorted([x for x in base.iterdir() if x.is_file() and x.suffix == '.json'])
    if jsons:
        return str(jsons[0])

    return None


def load_numpy_if_exists(path):
    p = Path(path)
    if p.exists():
        try:
            return np.load(str(p), allow_pickle=True)
        except Exception:
            return None
    return None


def display_turns(turn_data):
    # turn_data expected to contain 'last_play' list
    last_play = turn_data.get('last_play', [])
    if len(last_play) < 1:
        st.warning('No last_play entries found in snapshot')
        return

    # second last and last
    second_last = last_play[-2] if len(last_play) >= 2 else None
    last = last_play[-1]

    st.subheader('Play Turn')
    if second_last is None:
        st.info('No play turn available')
    else:
        # attempt to display prompt and action with safe access
        prompt = None
        try:
            prompt = second_last.get('prompt', [])[0].get('content')
        except Exception:
            prompt = None
        action = second_last.get('action') if second_last else None

        st.markdown('**Prompt:**')
        if prompt:
            st.text(prompt)
        else:
            st.text('N/A')

        st.markdown('**Action:**')
        st.json(action if action is not None else 'N/A')

    st.subheader('Challenge Turn')
    try:
        prompt = last.get('prompt', [])[0].get('content')
    except Exception:
        prompt = None
    action = last.get('action')

    st.markdown('**Prompt:**')
    if prompt:
        st.text(prompt)
    else:
        st.text('N/A')

    st.markdown('**Action:**')
    st.json(action if action is not None else 'N/A')


def main():
    st.title('BS Game Snapshot Dashboard')

    default_results = RESULTS_PATH
    result_path = st.text_input('Result path', value=default_results)

    game_seeds = list_game_seeds(result_path)
    if not game_seeds:
        st.warning('No game seeds found in the result path')
        return

    selected_game_seed = st.selectbox('Select game_seed', game_seeds)

    turns = list_turns_for_seed(result_path, selected_game_seed)
    if not turns:
        st.warning('No turns found for selected game seed')
        return

    selected_turn = st.selectbox('Select turn', turns)

    # seed entries within the selected turn
    seed_entries = list_seed_entries_for_turn(result_path, selected_game_seed, selected_turn)
    if not seed_entries:
        st.warning('No seed entries found inside selected turn')
        return

    # session state for navigation index
    if 'seed_idx' not in st.session_state or st.session_state.get('last_selected') != (selected_game_seed, selected_turn):
        st.session_state.seed_idx = 0
        st.session_state.last_selected = (selected_game_seed, selected_turn)

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button('Previous'):
            if st.session_state.seed_idx > 0:
                st.session_state.seed_idx -= 1
    with cols[1]:
        st.write(f'Index: {st.session_state.seed_idx} / {len(seed_entries)-1}')
    with cols[2]:
        if st.button('Next'):
            if st.session_state.seed_idx < len(seed_entries) - 1:
                st.session_state.seed_idx += 1

    # allow direct selection as well
    selected_turn_seed_index = st.selectbox('Select seed index', list(range(len(seed_entries))), index=st.session_state.seed_idx)
    # keep index in sync
    st.session_state.seed_idx = selected_turn_seed_index

    selected_turn_seed = seed_entries[st.session_state.seed_idx]

    st.markdown('---')
    st.write(f'Loading snapshot for `{selected_game_seed}` / `{selected_turn}` / `{selected_turn_seed}`')

    snapshot_file = find_snapshot_file(result_path, selected_game_seed, selected_turn, selected_turn_seed)
    if not snapshot_file:
        st.error('Could not find a snapshot JSON file for the selected entry')
        return

    try:
        with open(snapshot_file, 'r') as f:
            turn_data = json.load(f)
    except Exception as e:
        st.error(f'Failed to load snapshot JSON: {e}')
        return

    # try to load truthful/response trajectories if they exist under folder
    base_seed_folder = Path(result_path) / selected_game_seed / selected_turn / selected_turn_seed
    truthful_file = base_seed_folder / 'truthful_trajectory.npy'
    response_file = base_seed_folder / 'response_trajectory.npy'

    truthful_data = load_numpy_if_exists(truthful_file)
    response_data = load_numpy_if_exists(response_file)

    # We don't need these now
    #if truthful_data is not None:
    #    st.write('Truthful trajectory (sample)')
    #    st.write(truthful_data[:10])
    #if response_data is not None:
    #    st.write('Response trajectory (sample)')
    #    st.write(response_data[:10])

    # Display the requested fields
    display_turns(turn_data)


if __name__ == '__main__':
    main()
