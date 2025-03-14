import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Configuration
BASE_DIRS = ['db_1', 'db_2', 'db_3', 'db_4', 'db_5']
CHECK_DIR = 'db_13'
K = 5
M = 9
RESULTS_DIR = 'analysis_results'

CUT_OFF_TOP_P = 1e-4
os.makedirs(RESULTS_DIR, exist_ok=True)

def gather_original_data():
    """
    Collect original data and probabilities from BASE_DIRS.
    Since each fname is in exactly one base_dir, we'll store probabilities
    to pair with check data later.

    Returns:
      originals: dict mapping "filename" -> { 'data': ..., 'source': ... }
      orig_probs: dict with keys (fname, i, tok_id) and values = prob
    """
    originals = {}
    orig_probs = {}  # (fname, i, tok_id) -> probability from base_dir

    for base_dir in BASE_DIRS:
        print(f'Loading originals from {base_dir}...')
        base_dir_path = os.path.abspath(base_dir)
        if not os.path.isdir(base_dir_path):
            continue

        for fname in tqdm(os.listdir(base_dir_path)):
            full_path = os.path.join(base_dir_path, fname)
            if not os.path.isfile(full_path):
                continue
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
            except:
                continue

            originals[fname] = {
                'data': data,
                'source': base_dir
            }

            exec_data = data.get('execution_data', [])
            for i, token_info in enumerate(exec_data):
                for el in token_info['top_k']:
                    tok_id = el['id']
                    prob = el['prob']
                    orig_probs[(fname, i, tok_id)] = prob  # Only one prob per fname

    return originals, orig_probs

def calculate_perplexity(data_list):
    log_probs = [np.log(t['prob']) for t in data_list if t['prob'] > 0]
    return np.exp(-np.mean(log_probs)) if log_probs else float('inf')

def analyze_check_data(originals, orig_probs):
    """
    Process files in CHECK_DIR, compute epsilon against originals, and perform analyses.
    Epsilon is now |orig_prob - check_prob| for each (fname, i, tok_id).

    Returns:
      prob_diffs, ranking_data, margin_violations, perplexities, position_changes,
      local_epsilon (computed here)
    """
    prob_diffs = []
    ranking_data = {'preserved': 0, 'total': 0}
    margin_violations = 0
    perplexities = {'original': [], 'checked': []}
    position_changes = []
    local_epsilon = {}  # (fname, i, tok_id) -> |orig_prob - check_prob|

    check_dir_path = os.path.abspath(CHECK_DIR)
    if not os.path.isdir(check_dir_path):
        print(f"[Warning] CHECK_DIR={CHECK_DIR} doesn't exist.")
        return prob_diffs, ranking_data, margin_violations, perplexities, position_changes, local_epsilon

    print(f'Processing verification files in {CHECK_DIR}...')
    for check_file in tqdm(os.listdir(check_dir_path)):
        check_path = os.path.join(check_dir_path, check_file)
        if not os.path.isfile(check_path):
            continue

        # Parse original filename from check_file (e.g., "check_machine_someFile.json")
        parts = check_file.split('_')
        if len(parts) < 3:
            continue
        orig_key = '_'.join(parts[2:])  # e.g., "someFile.json" (include extension)

        original_entry = originals.get(orig_key)
        if not original_entry:
            continue

        # Load check data
        try:
            with open(check_path, 'r') as f:
                check_json = json.load(f)
        except:
            continue

        orig_exec_data = original_entry['data'].get('execution_data', [])
        check_exec_data = check_json.get('check_data', [])

        # Compute epsilon and other analyses
        for step_idx, (orig_t, check_t) in enumerate(zip(orig_exec_data, check_exec_data)):
            orig_topk_dict = {el['id']: el['prob'] for el in orig_t['top_k']}
            check_topk_dict = {el['id']: el['prob'] for el in check_t['top_k']}

            # _key = ('d0e7c29a451d0b0de3975e954c5c893828e314f309a06cd34b66a273f0e96505', 338, 66742)
            # if orig_key == _key[0] and step_idx == _key[1] and _key[2] in orig_topk_dict:
            #     print(f"orig_topk_dict = {orig_topk_dict}")
            #     print(f"check_topk_dict = {check_topk_dict}")

            # Compute epsilon for matching tokens
            for tok_id in set(orig_topk_dict.keys()) & set(check_topk_dict.keys()):
                orig_prob = orig_topk_dict[tok_id]
                check_prob = check_topk_dict[tok_id]
                local_epsilon[(orig_key, step_idx, tok_id)] = abs(orig_prob - check_prob)


            # Probability differences
            if orig_t['id'] == check_t['id']:
                prob_diffs.append(abs(orig_t['prob'] - check_t['prob']))

            # Ranking & margin checks
            orig_topk = [tk['id'] for tk in orig_t['top_k'][:K] if tk['prob'] > 1e-7]
            check_topkm = [tk['id'] for tk in check_t['top_k'][:(K + M)] if tk['prob'] > 1e-7]
            preserved_count = sum(1 for x in orig_topk if x in check_topkm)
            ranking_data['preserved'] += preserved_count
            ranking_data['total'] += len(orig_topk)

            if len(check_t['top_k']) >= (K + M):

                token_k = check_t['top_k'][K-1]
                token_km = check_t['top_k'][K + M]
                p_k = token_k['prob']
                p_km = token_km['prob']

                if p_k < CUT_OFF_TOP_P or p_km < CUT_OFF_TOP_P:
                    continue
                margin = p_k - p_km

                eps_k = local_epsilon.get((orig_key, step_idx, token_k['id']), 0.0)
                eps_km = local_epsilon.get((orig_key, step_idx, token_km['id']), 0.0)

                if margin < 2 * max(eps_k, eps_km) and margin > 1e-6 and max(eps_k, eps_km)> 1e-6: # if eps is zero, it doesn't make sense to compare
                    # print(f"Margin violation: {margin:.6f} <= 2 * max({eps_k:.6f}, {eps_km:.6f}) p_k={p_k:.6f} p_km={p_km:.6f} probs={[el['prob'] for el in check_t['top_k']]} ")
                    # print(f"(orig_key, step_idx, token_k['id']) = {(orig_key, step_idx, token_k['id'])}")
                    margin_violations += 1

            # Position changes
            orig_ranks = {tk['id']: idx for idx, tk in enumerate(orig_t['top_k'])}
            check_subset = [el for el in check_t['top_k'][:15] if el['prob'] > 1e-7]
            for check_rank, vt in enumerate(check_subset):
                if vt['id'] in orig_ranks:
                    position_changes.append((orig_ranks[vt['id']], check_rank))

        # Perplexity
        perplexities['original'].append(calculate_perplexity(orig_exec_data))
        perplexities['checked'].append(calculate_perplexity(check_exec_data))

    return prob_diffs, ranking_data, margin_violations, perplexities, position_changes, local_epsilon

def make_plots(prob_diffs, perplexities, position_changes, epsilon_stats=(0,0,0,0,0,0)):
    # [Unchanged plotting code]
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(prob_diffs, bins=50)
    y_max = max(counts)  # This gives you the maximum height of the histogram bars
    plt.xlabel('Absolute Probability Difference')
    plt.ylabel('Frequency')
    plt.title('Numerical Stability (Per-Token Differences)')
    eps_labels = ['Max', 'Min', 'Avg', '95th', '99th', '99.9th']
    for idx, val in enumerate(epsilon_stats):
        plt.axvline(val, color='r', linestyle='--')
        # plt.text(val, idx, f'{eps_labels[idx]}', rotation=90,  ha='right')
        # Draw the label near the top of the figure, offset a bit
        plt.annotate(
            f'{eps_labels[idx]}',
            xy=(val, y_max * ((idx+1)/len(eps_labels)) * 0.9),       # Where the arrow points
            xytext=(val+0.001, y_max * ((idx+1)/len(eps_labels)) * 0.95),  # Where the text is placed
            ha='right', va='bottom',     # Alignments for text
            arrowprops=dict(arrowstyle='->', color='red')
    )
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'probability_differences.svg'), format='svg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(perplexities['original'], perplexities['checked'], alpha=0.6)
    finite_o = [p for p in perplexities['original'] if np.isfinite(p)]
    finite_c = [p for p in perplexities['checked'] if np.isfinite(p)]
    if finite_o and finite_c:
        max_val = max(finite_o + finite_c)
        plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Original Perplexity')
    plt.ylabel('Checked Perplexity')
    plt.title('Perplexity Comparison')
    plt.savefig(os.path.join(RESULTS_DIR, 'perplexity_stability.svg'), format='svg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    # Filter position_changes to include only pairs where ranks differ
    changed_positions = [(orig, check) for orig, check in position_changes if orig != check]
    if changed_positions:  # Avoid empty data issues
        x_vals = [v[0] for v in changed_positions]
        y_vals = [v[1] for v in changed_positions]
        plt.hist2d(x_vals, y_vals, bins=(15, 15))
        plt.colorbar(label='Frequency')
        plt.xlabel('Original Rank')
        plt.ylabel('Checked Rank')
        plt.title('Rank Stability Heatmap (Changed Positions Only)')
        plt.savefig(os.path.join(RESULTS_DIR, 'ranking_stability_heatmap_changed.svg'), format='svg', bbox_inches='tight')
    else:
        print("No position changes detected; skipping heatmap.")
    plt.close()

if __name__ == '__main__':
    # Gather original data
    originals, orig_probs = gather_original_data()

    # Analyze check data and compute epsilon
    prob_diffs, ranking_data, margin_violations, perplexities, position_changes, local_epsilon = \
        analyze_check_data(originals, orig_probs)

    epsilon_values = list(local_epsilon.values())
    epsilon_stats = (
        max(epsilon_values) if epsilon_values else 0,
        min(epsilon_values) if epsilon_values else 0,
        np.mean(epsilon_values) if epsilon_values else 0,
        np.percentile(epsilon_values, 95) if epsilon_values else 0,
        np.percentile(epsilon_values, 99) if epsilon_values else 0,
        np.percentile(epsilon_values, 99.9) if epsilon_values else 0
    )

    # Make plots
    make_plots(prob_diffs, perplexities, position_changes, epsilon_stats)

    # Print summary
    print("\n===== RESULTS SUMMARY =====")
    print(f"\nLocal Epsilon Stats:")
    print(f"Max                 : {epsilon_stats[0]:.6f}")
    print(f"Min                 : {epsilon_stats[1]:.6f}")
    print(f"Avg                 : {epsilon_stats[2]:.6f}")
    print(f"95th percentile     : {epsilon_stats[3]:.6f}")
    print(f"99th percentile     : {epsilon_stats[4]:.6f}")
    print(f"99.9th percentile   : {epsilon_stats[5]:.6f}")

    if prob_diffs:
        print(f"\nTotal matched tokens for diff: {len(prob_diffs)}")
        print(f"Avg absolute diff: {np.mean(prob_diffs):.6f}")
        print(f"Max absolute diff: {np.max(prob_diffs):.6f}")
    else:
        print("\nNo probability differences computed.")

    total = ranking_data['total']
    preserved = ranking_data['preserved']
    preserve_rate = preserved / total if total else 0.0
    print(f"\nTop-{K} Preservation Rate: {preserve_rate:.2%}")
    print(f"Margin Violations: {margin_violations}")

    orig_ppls = perplexities['original']
    check_ppls = perplexities['checked']
    if orig_ppls and check_ppls:
        print(f"\nOriginal perplexity range: {min(orig_ppls):.2f} - {max(orig_ppls):.4f}")
        print(f"Checked  perplexity range: {min(check_ppls):.2f} - {max(check_ppls):.4f}")
    else:
        print("\nNo perplexity data found.")

    print(f"\nPlots saved to: {RESULTS_DIR}/\n")