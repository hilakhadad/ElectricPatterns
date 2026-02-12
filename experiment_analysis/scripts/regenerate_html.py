"""
Regenerate only the aggregate HTML report from existing per-house JSON files.

Usage:
    python regenerate_html.py <analysis_output_dir>

Example:
    python regenerate_html.py /home/hilakese/ElectricPatterns_new/experiment_analysis/OUTPUT/analysis_exp007_20260212_081553_20260212_150632
"""
import sys
import json
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from visualization.html_report import generate_html_report


def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_html.py <analysis_output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1]).resolve()
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

    # Extract experiment name from directory name
    experiment_name = output_dir.name.split("analysis_")[-1].rsplit("_", 2)[0] if "analysis_" in output_dir.name else output_dir.name

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
