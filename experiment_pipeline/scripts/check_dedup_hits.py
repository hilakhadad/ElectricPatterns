"""
Investigate duplicate matches — separates FALSE POSITIVES from real leakage.

Two phenomena exist:
  1. SAME-iteration "duplicates" — actually distinct events ±2min apart with
     similar magnitude (e.g., compressor cycles). These are FALSE POSITIVES
     in the dedup algorithm, NOT pipeline bugs.
  2. CROSS-iteration duplicates — the same physical event appearing in matches/
     of two different iterations. This indicates segmentation didn't fully
     extract the event, so the next iteration re-detected it.

Usage (on server):
    python scripts/check_dedup_hits.py --experiment /path/to/exp010_...
    python scripts/check_dedup_hits.py                          # auto-detect
    python scripts/check_dedup_hits.py --house 305              # single house deep-dive
    python scripts/check_dedup_hits.py --house 305 --max 5      # limit output
"""
import sys
import csv
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

import pandas as pd
from identification.session_grouper import load_all_matches

# Diagnostic-only constants (these are NOT used in the pipeline anymore —
# the pipeline now filters by duration instead of deduplicating by proximity)
DEDUP_TIME_TOLERANCE_MINUTES = 2
DEDUP_MAGNITUDE_TOLERANCE_W = 50


def find_latest_experiment(base: Path) -> Path:
    experiments_dir = base / "OUTPUT" / "experiments"
    if not experiments_dir.exists():
        return None
    dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and 'exp010' in d.name],
        key=lambda d: d.stat().st_mtime,
    )
    return dirs[-1] if dirs else None


def discover_houses(experiment_dir: Path) -> list:
    houses = set()
    for run_dir in experiment_dir.glob("run_*"):
        for house_dir in run_dir.glob("house_*"):
            hid = house_dir.name.replace("house_", "")
            if (house_dir / "matches").exists():
                houses.add(hid)
    return sorted(houses)


def find_duplicate_groups(all_matches: pd.DataFrame):
    """Find groups of events flagged as duplicates by the ±2min/±50W rule.

    Returns (groups, df) where each group is a list of row indices.
    Groups are classified as:
      - 'same_iter'  if ALL members share the same iteration
      - 'cross_iter' if members span 2+ iterations
    """
    if all_matches.empty:
        return [], pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(all_matches['on_start']):
        all_matches['on_start'] = pd.to_datetime(all_matches['on_start'])

    df = all_matches.sort_values(['phase', 'on_start']).reset_index(drop=True)

    visited = set()
    duplicate_groups = []

    for i in range(len(df)):
        if i in visited:
            continue

        row_i = df.iloc[i]
        group = [i]
        visited.add(i)

        for j in range(i + 1, len(df)):
            if j in visited:
                continue
            row_j = df.iloc[j]

            if row_j['phase'] != row_i['phase']:
                if row_j['phase'] > row_i['phase']:
                    break
                continue

            time_diff = abs((row_j['on_start'] - row_i['on_start']).total_seconds() / 60)
            if time_diff > DEDUP_TIME_TOLERANCE_MINUTES * 2:
                break

            mag_diff = abs(abs(row_j['on_magnitude']) - abs(row_i['on_magnitude']))
            if time_diff <= DEDUP_TIME_TOLERANCE_MINUTES and mag_diff <= DEDUP_MAGNITUDE_TOLERANCE_W:
                group.append(j)
                visited.add(j)

        if len(group) >= 2:
            iters = set(int(df.iloc[idx]['iteration']) for idx in group)
            group_type = 'same_iter' if len(iters) == 1 else 'cross_iter'
            duplicate_groups.append((group, group_type))

    return duplicate_groups, df


def analyze_house(experiment_dir, house_id, threshold_schedule, max_examples=None):
    """Deep analysis of one house's duplicates, separated by type."""
    all_matches = load_all_matches(experiment_dir, house_id, threshold_schedule)
    if all_matches.empty:
        return {
            'house_id': house_id, 'total': 0,
            'same_iter_groups': 0, 'same_iter_events': 0,
            'cross_iter_groups': 0, 'cross_iter_events': 0,
            'details_same': [], 'details_cross': [],
            'iter_pairs': {}, 'tag_pairs_cross': {},
        }

    groups_typed, df = find_duplicate_groups(all_matches)

    details_same = []
    details_cross = []

    for group_indices, group_type in groups_typed:
        members = []
        for idx in group_indices:
            row = df.iloc[idx]
            members.append({
                'iteration': int(row['iteration']),
                'threshold': int(row['threshold']),
                'phase': row['phase'],
                'on_start': str(row['on_start']),
                'on_magnitude': round(float(row['on_magnitude']), 1),
                'off_magnitude': round(float(row.get('off_magnitude', 0)), 1) if 'off_magnitude' in row.index else None,
                'duration': round(float(row['duration']), 1) if 'duration' in row.index else None,
                'tag': row.get('tag', '?'),
                'on_event_id': str(row.get('on_event_id', '?')),
            })
        if group_type == 'same_iter':
            details_same.append(members)
        else:
            details_cross.append(members)

    same_extra = sum(len(g) - 1 for g, t in groups_typed if t == 'same_iter')
    cross_extra = sum(len(g) - 1 for g, t in groups_typed if t == 'cross_iter')

    # Iteration pair stats (cross-iter only)
    iter_pairs = {}
    for group_members in details_cross:
        iters = sorted(set(m['iteration'] for m in group_members))
        key = tuple(iters)
        iter_pairs[key] = iter_pairs.get(key, 0) + 1

    # Tag stats (cross-iter only — same-iter tags are always identical)
    tag_pairs_cross = {}
    for group_members in details_cross:
        tags = sorted(set(m['tag'] for m in group_members))
        key = ' vs '.join(tags)
        tag_pairs_cross[key] = tag_pairs_cross.get(key, 0) + 1

    # Duration category stats for same-iter false positives
    same_iter_durations = []
    for group_members in details_same:
        for m in group_members:
            if m['duration'] is not None:
                same_iter_durations.append(m['duration'])

    # Cross-iter: check if PARTIAL tag involved
    cross_partial_count = 0
    for group_members in details_cross:
        if any('PARTIAL' in m['tag'] for m in group_members):
            cross_partial_count += 1

    cross_tail_count = 0
    for group_members in details_cross:
        if any('TAIL' in str(m.get('tag', '')) or 'EXTENDED' in str(m.get('tag', ''))
               for m in group_members):
            cross_tail_count += 1

    if max_examples is not None:
        details_same = details_same[:max_examples]
        details_cross = details_cross[:max_examples]

    return {
        'house_id': house_id,
        'total': len(all_matches),
        'same_iter_groups': len([g for g, t in groups_typed if t == 'same_iter']),
        'same_iter_events': same_extra,
        'cross_iter_groups': len([g for g, t in groups_typed if t == 'cross_iter']),
        'cross_iter_events': cross_extra,
        'details_same': details_same,
        'details_cross': details_cross,
        'iter_pairs': iter_pairs,
        'tag_pairs_cross': tag_pairs_cross,
        'same_iter_durations': same_iter_durations,
        'cross_partial_count': cross_partial_count,
        'cross_tail_count': cross_tail_count,
    }


def print_house_report(result):
    """Print detailed forensics for one house, separated by type."""
    hid = result['house_id']
    total_groups = result['same_iter_groups'] + result['cross_iter_groups']
    total_extra = result['same_iter_events'] + result['cross_iter_events']

    print(f"\n{'='*70}")
    print(f"HOUSE {hid}: {result['total']} total matches")
    print(f"{'='*70}")

    if total_groups == 0:
        print("  No duplicates found.")
        return

    # ── Summary ──
    print(f"\n  SUMMARY:")
    print(f"    Same-iteration FALSE POSITIVES: {result['same_iter_groups']} groups "
          f"({result['same_iter_events']} extra events)")
    print(f"    Cross-iteration REAL duplicates: {result['cross_iter_groups']} groups "
          f"({result['cross_iter_events']} extra events)")
    print(f"    TOTAL: {total_groups} groups ({total_extra} extra events)")

    # ── Same-iteration details ──
    if result['same_iter_groups'] > 0:
        print(f"\n  {'─'*60}")
        print(f"  SAME-ITERATION FALSE POSITIVES ({result['same_iter_groups']} groups)")
        print(f"  These are DISTINCT events ±2min apart, NOT duplicates.")
        print(f"  {'─'*60}")

        # Duration breakdown
        durs = result.get('same_iter_durations', [])
        if durs:
            spikes = sum(1 for d in durs if d <= 2)
            quick = sum(1 for d in durs if 2 < d < 5)
            medium = sum(1 for d in durs if 5 <= d <= 30)
            extended = sum(1 for d in durs if d > 30)
            print(f"\n    Duration breakdown of false-positive events:")
            print(f"      SPIKE (<=2min): {spikes}  |  QUICK (2-5min): {quick}  |  "
                  f"MEDIUM (5-30min): {medium}  |  EXTENDED (>30min): {extended}")

        for i, group_members in enumerate(result['details_same']):
            print(f"\n    --- FP Group {i+1} ---")
            for m in sorted(group_members, key=lambda x: x['on_start']):
                dur_str = f"{m['duration']:.0f}min" if m['duration'] else "?"
                off_str = f"off={m['off_magnitude']:.0f}W" if m['off_magnitude'] else ""
                print(f"      iter={m['iteration']} TH={m['threshold']}W | "
                      f"{m['phase']} {m['on_start']} | "
                      f"on={m['on_magnitude']:.0f}W {off_str} | "
                      f"{dur_str} | {m['tag']}")
                if m.get('on_event_id', '?') != '?':
                    print(f"        event_id: {m['on_event_id']}")

    # ── Cross-iteration details ──
    if result['cross_iter_groups'] > 0:
        print(f"\n  {'─'*60}")
        print(f"  CROSS-ITERATION DUPLICATES ({result['cross_iter_groups']} groups)")
        print(f"  Same physical event in matches/ of different iterations.")
        print(f"  {'─'*60}")

        # Iteration pairs
        print(f"\n    Iteration overlap:")
        for iters, count in sorted(result['iter_pairs'].items()):
            iter_str = ' & '.join(f"iter{i}" for i in iters)
            print(f"      {iter_str}: {count} groups")

        # Cause indicators
        cp = result.get('cross_partial_count', 0)
        ct = result.get('cross_tail_count', 0)
        print(f"\n    Cause indicators:")
        print(f"      Groups with PARTIAL tag: {cp}/{result['cross_iter_groups']}")
        print(f"      Groups with EXTENDED tag: {ct}/{result['cross_iter_groups']}")

        # Tag pairs
        if result.get('tag_pairs_cross'):
            print(f"\n    Tag comparison (cross-iter):")
            for tags, count in sorted(result['tag_pairs_cross'].items(), key=lambda x: -x[1])[:10]:
                print(f"      {tags}: {count} groups")

        for i, group_members in enumerate(result['details_cross']):
            print(f"\n    --- Cross Group {i+1} ---")
            for m in sorted(group_members, key=lambda x: x['iteration']):
                dur_str = f"{m['duration']:.0f}min" if m['duration'] else "?"
                off_str = f"off={m['off_magnitude']:.0f}W" if m['off_magnitude'] else ""
                print(f"      iter={m['iteration']} TH={m['threshold']}W | "
                      f"{m['phase']} {m['on_start']} | "
                      f"on={m['on_magnitude']:.0f}W {off_str} | "
                      f"{dur_str} | {m['tag']}")
                if m.get('on_event_id', '?') != '?':
                    print(f"        event_id: {m['on_event_id']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Investigate duplicate matches — separates false positives from real leakage"
    )
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--house', type=str, default=None, help="Single house deep-dive")
    parser.add_argument('--max', type=int, default=20, help="Max example groups per type per house")
    parser.add_argument('--csv', type=str, default=None, help="Write summary CSV to this path")
    args = parser.parse_args()

    pipeline_root = script_dir.parent

    if args.experiment:
        experiment_dir = Path(args.experiment)
    else:
        experiment_dir = find_latest_experiment(pipeline_root)

    if experiment_dir is None or not experiment_dir.exists():
        print("ERROR: No experiment directory found. Use --experiment <path>")
        sys.exit(1)

    print(f"Experiment: {experiment_dir}")
    print(f"Dedup tolerance: ±{DEDUP_TIME_TOLERANCE_MINUTES} min, ±{DEDUP_MAGNITUDE_TOLERANCE_W}W")

    threshold_schedule = [2000, 1500, 1100, 800]

    # Single house mode
    if args.house:
        result = analyze_house(experiment_dir, args.house, threshold_schedule, max_examples=args.max)
        print_house_report(result)
        return

    # All houses mode
    houses = discover_houses(experiment_dir)
    print(f"Found {len(houses)} houses\n")

    all_results = []
    total_same = 0
    total_cross = 0
    total_matches = 0

    header = (f"{'House':>8}  {'Total':>7}  {'SameIter':>8}  {'CrossIter':>9}  "
              f"{'FP%':>6}  {'Real%':>6}")
    print(header)
    print("-" * len(header))

    for hid in houses:
        result = analyze_house(experiment_dir, hid, threshold_schedule, max_examples=0)
        all_results.append(result)
        total_matches += result['total']
        total_same += result['same_iter_events']
        total_cross += result['cross_iter_events']

        fp_pct = (result['same_iter_events'] / result['total'] * 100) if result['total'] > 0 else 0
        real_pct = (result['cross_iter_events'] / result['total'] * 100) if result['total'] > 0 else 0
        marker = ""
        if result['cross_iter_events'] > 0:
            marker = "  <<<< REAL"
        elif result['same_iter_events'] > 0:
            marker = "  (FP only)"
        print(f"{hid:>8}  {result['total']:>7}  {result['same_iter_events']:>8}  "
              f"{result['cross_iter_events']:>9}  {fp_pct:>5.1f}%  {real_pct:>5.1f}%{marker}")

    print("-" * len(header))
    fp_pct = (total_same / total_matches * 100) if total_matches > 0 else 0
    real_pct = (total_cross / total_matches * 100) if total_matches > 0 else 0
    print(f"{'TOTAL':>8}  {total_matches:>7}  {total_same:>8}  {total_cross:>9}  "
          f"{fp_pct:>5.1f}%  {real_pct:>5.1f}%")

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS:")
    print(f"  Same-iteration FALSE POSITIVES: {total_same} events ({fp_pct:.1f}%)")
    print(f"    -> Distinct events ±2min apart, NOT a pipeline bug")
    print(f"    -> Fix: exclude same-iteration pairs from dedup")
    print(f"  Cross-iteration REAL duplicates: {total_cross} events ({real_pct:.1f}%)")
    print(f"    -> Same physical event in 2+ iterations' matches/")
    print(f"    -> Causes: segmentation skip, PARTIAL remainders, tail shifts")
    print(f"    -> Dedup correctly handles these (keeps highest iteration)")
    print(f"{'='*60}")

    # Aggregate cross-iter stats
    houses_with_cross = [r for r in all_results if r['cross_iter_events'] > 0]
    if houses_with_cross:
        all_iter_pairs = {}
        for r in houses_with_cross:
            for iters, count in r['iter_pairs'].items():
                all_iter_pairs[iters] = all_iter_pairs.get(iters, 0) + count

        print(f"\n  Cross-iteration overlap (all houses):")
        for iters, count in sorted(all_iter_pairs.items(), key=lambda x: -x[1]):
            iter_str = ' & '.join(f"iter{i}" for i in iters)
            print(f"    {iter_str}: {count} groups")

        total_partial = sum(r.get('cross_partial_count', 0) for r in houses_with_cross)
        total_cross_groups = sum(r['cross_iter_groups'] for r in houses_with_cross)
        print(f"\n  Cross-iter groups involving PARTIAL: {total_partial}/{total_cross_groups}")

        worst = sorted(houses_with_cross, key=lambda x: -x['cross_iter_events'])[:3]
        print(f"\n  Worst 3 houses (cross-iter, for --house deep-dive):")
        for r in worst:
            print(f"    House {r['house_id']}: {r['cross_iter_events']} real duplicates out of {r['total']}")

    # CSV export
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'house_id', 'total_matches',
                'same_iter_groups', 'same_iter_extra', 'same_iter_pct',
                'cross_iter_groups', 'cross_iter_extra', 'cross_iter_pct',
            ])
            for r in all_results:
                fp_pct = (r['same_iter_events'] / r['total'] * 100) if r['total'] > 0 else 0
                real_pct = (r['cross_iter_events'] / r['total'] * 100) if r['total'] > 0 else 0
                writer.writerow([
                    r['house_id'], r['total'],
                    r['same_iter_groups'], r['same_iter_events'], f"{fp_pct:.1f}",
                    r['cross_iter_groups'], r['cross_iter_events'], f"{real_pct:.1f}",
                ])
        print(f"\nCSV written to: {args.csv}")


if __name__ == '__main__':
    main()
