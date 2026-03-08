"""
Run all normalization experiments and generate full reports for comparison.

Runs 5 experiments (1 baseline + 4 normalization methods) on the same houses,
generates all 3 report types for each, and produces an HTML index page
mapping every report to its experiment.

Experiments:
  exp015_hole_repair    — Baseline (no normalization)
  exp016_ma_detrend     — Moving average detrending (2h window)
  exp017_phase_balance  — Phase baseline equalization
  exp018_mad_clean      — MAD-based outlier cleaning
  exp019_combined_norm  — Combined (MA detrend + phase balance + MAD clean)

Output structure:
  OUTPUT/experiments/norm_comparison_{timestamp}/
    ├── exp015_hole_repair/
    │   ├── run_0/ ... run_3/   (pipeline output)
    │   ├── device_sessions/
    │   └── reports/
    │       ├── house_report.html           (pre-analysis aggregate)
    │       ├── house_reports/              (per-house pre-analysis)
    │       ├── segregation_report.html     (M1 aggregate)
    │       ├── segregation_reports/        (per-house M1)
    │       ├── identification_report.html  (M2 aggregate)
    │       └── identification_reports/     (per-house M2)
    ├── exp016_ma_detrend/
    │   └── reports/ ...
    ├── ...
    ├── index.html                          (comparison index page)
    └── timing_summary.csv                  (timing for all experiments)

Usage:
    python scripts/run_normalization_comparison.py --houses 221,305
    python scripts/run_normalization_comparison.py --shortest 5
    python scripts/run_normalization_comparison.py --houses 221 --experiments exp015,exp016
"""
import sys
import os
import time
import subprocess
import csv
import argparse
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
EXPERIMENT_PIPELINE = PROJECT_ROOT / "experiment_pipeline"
DATA_DIR = PROJECT_ROOT / "INPUT" / "HouseholdData"
PYTHON = sys.executable

# ── Experiment definitions ───────────────────────────────────────────────
# Maps experiment name → normalization method for pre-analysis
EXPERIMENTS = [
    {
        'name': 'exp015_hole_repair',
        'label': 'Baseline (no normalization)',
        'normalize': 'none',
        'color': '#6c757d',
    },
    {
        'name': 'exp016_ma_detrend',
        'label': 'MA Detrending (2h window)',
        'normalize': 'ma_detrend',
        'color': '#007bff',
    },
    {
        'name': 'exp017_phase_balance',
        'label': 'Phase Balancing',
        'normalize': 'phase_balance',
        'color': '#28a745',
    },
    {
        'name': 'exp018_mad_clean',
        'label': 'MAD Outlier Cleaning',
        'normalize': 'mad_clean',
        'color': '#fd7e14',
    },
    {
        'name': 'exp019_combined_norm',
        'label': 'Combined (MA + Phase + MAD)',
        'normalize': 'combined',
        'color': '#6f42c1',
    },
]


def get_houses_sorted_by_months():
    """Get house IDs sorted by number of monthly pkl files (fewest first)."""
    houses = []
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            pkl_files = list(folder.glob("*.pkl"))
            if pkl_files:
                total_size = sum(f.stat().st_size for f in pkl_files)
                houses.append((folder.name, len(pkl_files), total_size))
    houses.sort(key=lambda x: (x[1], x[2]))
    return houses


def run_command(args, cwd, label=""):
    """Run a subprocess and stream output. Returns exit code."""
    print(f"\n{'-'*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  Command: {' '.join(str(a) for a in args)}", flush=True)
    print(f"{'-'*60}", flush=True)
    result = subprocess.run(args, cwd=str(cwd))
    return result.returncode


def run_single_experiment(exp_config: dict, house_ids: list,
                          comparison_dir: Path, timing_rows: list):
    """Run one full experiment: pre-analysis + pipeline + all reports."""
    exp_name = exp_config['name']
    normalize = exp_config['normalize']
    exp_output = comparison_dir / exp_name
    reports_dir = exp_output / "reports"
    os.makedirs(reports_dir, exist_ok=True)

    exp_start = time.time()

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {exp_config['label']}")
    print(f"  Normalization: {normalize}")
    print(f"  Output: {exp_output}")
    print(f"{'#'*60}")

    # ── Phase 1: House pre-analysis (with matching normalization) ─────
    print(f"\n{'='*60}")
    print(f"  Phase 1: PRE-ANALYSIS (normalization={normalize})")
    print(f"{'='*60}")

    pre_analysis_cmd = [
        PYTHON, "scripts/run_analysis.py",
        "--input-dir", str(DATA_DIR),
        "--houses", ",".join(house_ids),
        "--output-dir", str(reports_dir),
        "--publish", "house",
        "--normalize", normalize,
    ]
    run_command(
        pre_analysis_cmd,
        cwd=PROJECT_ROOT / "house_analysis",
        label=f"Pre-analysis: {len(house_ids)} houses (norm={normalize})",
    )

    # ── Phase 2: Pipeline + per-house reports ────────────────────────
    successful_houses = []
    failed_houses = []

    for i, house_id in enumerate(house_ids, 1):
        print(f"\n{'='*60}")
        print(f"  Phase 2: PIPELINE — House {house_id} ({i}/{len(house_ids)})")
        print(f"{'='*60}")

        house_start = time.time()

        exit_code = run_command(
            [PYTHON, "-u", "scripts/test_single_house.py",
             "--house_id", house_id,
             "--experiment_name", exp_name,
             "--output_path", str(exp_output),
             "--skip_visualization"],
            cwd=EXPERIMENT_PIPELINE,
            label=f"Pipeline [{exp_name}]: house {house_id}",
        )

        house_elapsed = time.time() - house_start
        mins = int(house_elapsed // 60)
        secs = int(house_elapsed % 60)

        if exit_code == 0:
            successful_houses.append(house_id)
            status = "OK"

            # Per-house segregation report
            run_command(
                [PYTHON, "scripts/run_dynamic_report.py",
                 "--experiment", str(exp_output),
                 "--houses", house_id,
                 "--output-dir", str(reports_dir),
                 "--publish", "segregation"],
                cwd=PROJECT_ROOT / "disaggregation_analysis",
                label=f"Segregation report [{exp_name}]: house {house_id}",
            )

            # Per-house identification report
            run_command(
                [PYTHON, "scripts/run_identification_report.py",
                 "--experiment", str(exp_output),
                 "--houses", house_id,
                 "--output-dir", str(reports_dir),
                 "--publish", "identification"],
                cwd=PROJECT_ROOT / "identification_analysis",
                label=f"Identification report [{exp_name}]: house {house_id}",
            )
        else:
            failed_houses.append(house_id)
            status = f"FAIL(exit={exit_code})"

        timing_rows.append({
            'experiment': exp_name,
            'house_id': house_id,
            'elapsed_seconds': f"{house_elapsed:.0f}",
            'elapsed_human': f"{mins}m {secs}s",
            'status': status,
        })

        print(f"  House {house_id} [{exp_name}]: {status} ({mins}m {secs}s)")

    # ── Phase 3: Aggregate reports ───────────────────────────────────
    if len(successful_houses) > 1:
        print(f"\n{'='*60}")
        print(f"  Phase 3: AGGREGATE REPORTS ({len(successful_houses)} houses)")
        print(f"{'='*60}")

        run_command(
            [PYTHON, "scripts/run_dynamic_report.py",
             "--experiment", str(exp_output),
             "--output-dir", str(reports_dir),
             "--publish", "segregation"],
            cwd=PROJECT_ROOT / "disaggregation_analysis",
            label=f"Aggregate segregation [{exp_name}]",
        )

        run_command(
            [PYTHON, "scripts/run_identification_report.py",
             "--experiment", str(exp_output),
             "--output-dir", str(reports_dir),
             "--publish", "identification"],
            cwd=PROJECT_ROOT / "identification_analysis",
            label=f"Aggregate identification [{exp_name}]",
        )

    exp_elapsed = time.time() - exp_start
    exp_mins = int(exp_elapsed // 60)
    exp_secs = int(exp_elapsed % 60)

    return {
        'name': exp_name,
        'label': exp_config['label'],
        'normalize': normalize,
        'color': exp_config['color'],
        'successful': successful_houses,
        'failed': failed_houses,
        'elapsed': f"{exp_mins}m {exp_secs}s",
        'reports_dir': str(reports_dir),
    }


def generate_index_html(comparison_dir: Path, results: list, house_ids: list):
    """Generate an HTML index page linking all experiment reports."""
    rows = ""
    for r in results:
        exp_name = r['name']
        n_ok = len(r['successful'])
        n_fail = len(r['failed'])
        color = r['color']

        # Build report links
        links = []
        reports_rel = f"{exp_name}/reports"

        house_report = comparison_dir / exp_name / "reports" / "house_report.html"
        if house_report.exists():
            links.append(f'<a href="{reports_rel}/house_report.html">Pre-analysis</a>')

        seg_report = comparison_dir / exp_name / "reports" / "segregation_report.html"
        if seg_report.exists():
            links.append(f'<a href="{reports_rel}/segregation_report.html">Segregation</a>')

        ident_report = comparison_dir / exp_name / "reports" / "identification_report.html"
        if ident_report.exists():
            links.append(f'<a href="{reports_rel}/identification_report.html">Identification</a>')

        links_html = " &nbsp;|&nbsp; ".join(links) if links else "(no reports)"

        # Per-house links
        per_house_links = []
        for house_id in house_ids:
            house_links = []
            pre_path = comparison_dir / exp_name / "reports" / "house_reports" / f"house_{house_id}.html"
            if pre_path.exists():
                house_links.append(f'<a href="{reports_rel}/house_reports/house_{house_id}.html">Pre</a>')
            seg_path = comparison_dir / exp_name / "reports" / "segregation_reports" / f"house_{house_id}.html"
            if seg_path.exists():
                house_links.append(f'<a href="{reports_rel}/segregation_reports/house_{house_id}.html">Seg</a>')
            ident_path = comparison_dir / exp_name / "reports" / "identification_reports" / f"house_{house_id}.html"
            if ident_path.exists():
                house_links.append(f'<a href="{reports_rel}/identification_reports/house_{house_id}.html">Ident</a>')
            if house_links:
                per_house_links.append(
                    f'<span style="margin-right:12px;"><b>{house_id}</b>: '
                    f'{" | ".join(house_links)}</span>'
                )

        per_house_html = "<br>".join(per_house_links) if per_house_links else ""

        status_color = "#28a745" if n_fail == 0 else "#dc3545"

        rows += f"""
        <tr>
            <td style="padding:10px 14px;border-bottom:1px solid #dee2e6;">
                <span style="display:inline-block;width:12px;height:12px;
                    border-radius:50%;background:{color};margin-right:8px;"></span>
                <b>{exp_name}</b><br>
                <small style="color:#666;">{r['label']}</small>
            </td>
            <td style="padding:10px 14px;border-bottom:1px solid #dee2e6;">
                <code>{r['normalize']}</code>
            </td>
            <td style="padding:10px 14px;border-bottom:1px solid #dee2e6;
                color:{status_color};">
                {n_ok} OK{f', {n_fail} failed' if n_fail else ''}
            </td>
            <td style="padding:10px 14px;border-bottom:1px solid #dee2e6;">{r['elapsed']}</td>
            <td style="padding:10px 14px;border-bottom:1px solid #dee2e6;">
                <b>Aggregate:</b> {links_html}
            </td>
        </tr>
        <tr>
            <td colspan="5" style="padding:6px 14px 14px 40px;
                border-bottom:2px solid #adb5bd;font-size:0.9em;">
                <b>Per-house:</b><br>{per_house_html}
            </td>
        </tr>
        """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Normalization Comparison — {timestamp}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
    h1 {{ color: #333; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
            border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
    th {{ background: #343a40; color: white; padding: 12px 14px; text-align: left; }}
    a {{ color: #007bff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .summary {{ background: white; padding: 16px 20px; border-radius: 8px;
               margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
</style>
</head><body>
<h1>Normalization Comparison</h1>
<div class="summary">
    <p><b>Generated:</b> {timestamp}</p>
    <p><b>Houses:</b> {', '.join(house_ids)}</p>
    <p><b>Experiments:</b> {len(results)}</p>
</div>
<table>
<tr>
    <th>Experiment</th>
    <th>Normalization</th>
    <th>Status</th>
    <th>Time</th>
    <th>Aggregate Reports</th>
</tr>
{rows}
</table>
</body></html>"""

    index_path = comparison_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nIndex page: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all normalization experiments and generate comparison reports"
    )
    parser.add_argument("--houses", type=str, default=None,
                        help="Comma-separated house IDs")
    parser.add_argument("--shortest", type=int, default=None,
                        help="Run N shortest houses (by month count)")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment names to run "
                             "(default: all 5). E.g. exp015,exp016")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Override output directory name "
                             "(default: norm_comparison_{timestamp})")
    args = parser.parse_args()

    # ── Determine house list ─────────────────────────────────────────
    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]
    elif args.shortest:
        all_houses = get_houses_sorted_by_months()
        house_ids = [h[0] for h in all_houses[:args.shortest]]
        print(f"Selected {args.shortest} shortest houses:")
        for name, n_months, size in all_houses[:args.shortest]:
            print(f"  House {name}: {n_months} months, {size/1024/1024:.1f} MB")
    else:
        # All houses
        all_houses = get_houses_sorted_by_months()
        house_ids = [h[0] for h in all_houses]
        print(f"Running on ALL {len(house_ids)} houses (sorted by size)")

    # ── Determine experiments to run ─────────────────────────────────
    if args.experiments:
        exp_names = [e.strip() for e in args.experiments.split(',')]
        experiments = [e for e in EXPERIMENTS if e['name'] in exp_names]
        if not experiments:
            print(f"ERROR: No matching experiments for: {exp_names}")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)
    else:
        experiments = EXPERIMENTS

    # ── Set up output directory ──────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = args.output_name or f"norm_comparison_{timestamp}"
    comparison_dir = EXPERIMENT_PIPELINE / "OUTPUT" / "experiments" / dir_name
    os.makedirs(comparison_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("NORMALIZATION COMPARISON RUNNER")
    print("=" * 60)
    print(f"Houses:      {len(house_ids)} — {house_ids}")
    print(f"Experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['name']} ({exp['label']})")
    print(f"Output:      {comparison_dir}")
    print("=" * 60)

    # ── Run each experiment ──────────────────────────────────────────
    total_start = time.time()
    timing_rows = []
    results = []

    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT {i}/{len(experiments)}: {exp_config['name']}")
        print(f"{'#'*60}")

        result = run_single_experiment(
            exp_config, house_ids, comparison_dir, timing_rows
        )
        results.append(result)

    # ── Save timing CSV ──────────────────────────────────────────────
    timing_file = comparison_dir / "timing_summary.csv"
    with open(timing_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'house_id', 'elapsed_seconds', 'elapsed_human', 'status'
        ])
        writer.writeheader()
        writer.writerows(timing_rows)

    # ── Generate index HTML ──────────────────────────────────────────
    generate_index_html(comparison_dir, results, house_ids)

    # ── Final summary ────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_mins = int((total_elapsed % 3600) // 60)
    total_secs = int(total_elapsed % 60)

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Total time:   {total_hours}h {total_mins}m {total_secs}s")
    print(f"Output:       {comparison_dir}")
    print(f"Index:        {comparison_dir / 'index.html'}")
    print(f"Timing:       {timing_file}")
    print()

    for r in results:
        status = f"{len(r['successful'])} OK"
        if r['failed']:
            status += f", {len(r['failed'])} FAILED"
        print(f"  {r['name']:30s}  {status:15s}  {r['elapsed']}")

    print(f"\n{'='*60}")
    print(f"Open the index page to see all reports:")
    print(f"  {comparison_dir / 'index.html'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
