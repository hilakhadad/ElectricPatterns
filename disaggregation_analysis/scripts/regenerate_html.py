"""
Regenerate only the aggregate HTML report from existing per-house JSON files.

Automatically loads pre-analysis quality scores from house_analysis output.

Usage:
    python regenerate_html.py <analysis_output_dir>
    python regenerate_html.py <analysis_output_dir> --pre-analysis <house_analysis_run_dir>

Example:
    python regenerate_html.py /path/to/experiment_analysis/OUTPUT/analysis_exp007_...
"""
import sys
import json
import argparse
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from visualization.html_report import generate_html_report
from reports.aggregate_report import load_pre_analysis_scores


def main():
    parser = argparse.ArgumentParser(description='Regenerate HTML report from existing JSON data')
    parser.add_argument('output_dir', type=str, help='Path to analysis output directory')
    parser.add_argument('--pre-analysis', type=str, default=None,
                        help='Path to house_analysis run directory (auto-detected if not specified)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    house_reports_dir = output_dir / "house_reports"

    if not house_reports_dir.exists():
        print(f"Error: house_reports directory not found: {house_reports_dir}")
        sys.exit(1)

    # Load all per-house JSON analysis files
    json_files = sorted(house_reports_dir.glob("house_*_analysis.json"))
    if not json_files:
        print(f"Error: No house_*_analysis.json files found in {house_reports_dir}")
        sys.exit(1)

    print(f"Loading {len(json_files)} house analysis files...")
    analyses = []
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            analyses.append(analysis)
        except Exception as e:
            print(f"  Warning: Failed to load {jf.name}: {e}")

    print(f"Loaded {len(analyses)} analyses")

    # Load and inject pre-analysis quality scores
    pre_analysis_path = None
    if args.pre_analysis:
        pre_analysis_path = Path(args.pre_analysis).resolve()
    else:
        # Auto-detect: look for house_analysis/OUTPUT/run_* relative to project
        # Try multiple possible locations
        project_root = SCRIPT_DIR.parent.parent
        house_analysis_output = project_root / "house_analysis" / "OUTPUT"
        if house_analysis_output.exists():
            run_dirs = sorted(
                [d for d in house_analysis_output.iterdir()
                 if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if run_dirs:
                pre_analysis_path = run_dirs[0]
                print(f"Auto-detected house_analysis output: {pre_analysis_path.name}")

    if pre_analysis_path and pre_analysis_path.exists():
        pre_analysis_scores = load_pre_analysis_scores(pre_analysis_path)
        if pre_analysis_scores:
            matched = 0
            for analysis in analyses:
                house_id = str(analysis.get('house_id', ''))
                if house_id in pre_analysis_scores:
                    analysis['pre_analysis_quality_score'] = pre_analysis_scores[house_id]
                    matched += 1
            print(f"Injected pre-analysis quality scores for {matched}/{len(analyses)} houses")
    else:
        print("Warning: No pre-analysis scores found. Tier filtering will show all as 'Unknown'.")

    # Extract experiment name from directory name
    dir_name = output_dir.name
    if "analysis_" in dir_name:
        experiment_name = dir_name.split("analysis_", 1)[-1].rsplit("_", 2)[0]
    else:
        experiment_name = dir_name

    # Generate HTML report
    html_path = output_dir / "report.html"
    print(f"Generating HTML report...")
    generate_html_report(
        analyses,
        str(html_path),
        title=f"Experiment Analysis: {experiment_name}"
    )
    print(f"Saved: {html_path}")


if __name__ == "__main__":
    main()
