"""
HTML report generator for device identification (Module 2).

Generates stand-alone HTML reports per house and aggregate reports.
Follows the same patterns as dynamic_html_report.py (inline CSS, Plotly CDN).

Sections:
  1. Session Overview Dashboard — pie/bar of device types, summary cards
  2. Confidence Distribution — histogram + tier breakdown
  3. Boiler Analysis — daily pattern, monthly consistency, magnitude
  4. AC Analysis — central vs regular, seasonal, cycles
  5. Temporal Patterns — hour x day-of-week heatmap
  6. Classification Quality — 5 metrics + flags (from classification_quality.py)
  7. Unclassified Analysis — power/duration distributions
  8. Device Activations Detail — per-event sortable tables
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from metrics.classification_quality import calculate_classification_quality
from metrics.confidence_scoring import calculate_confidence_scores
from metrics.population_statistics import compute_population_statistics

from visualization.identification_charts import (
    create_session_overview,
    create_confidence_overview,
    create_boiler_analysis,
    create_ac_analysis,
    create_temporal_heatmap,
    create_unclassified_analysis,
    create_device_activations_detail,
    create_spike_analysis,
)
from visualization.classification_charts import create_quality_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-analysis quality helpers (mirrors disaggregation_analysis logic)
# ---------------------------------------------------------------------------

def _load_pre_analysis_scores(house_analysis_path) -> Dict[str, Any]:
    """
    Load quality scores from house_analysis output.

    Supports:
    - Directory containing per-house JSON files (per_house/ or direct)
    - Single JSON file with list of analyses or 'analyses' key
    """
    import json as _json
    scores = {}
    house_analysis_path = Path(house_analysis_path)

    if not house_analysis_path.exists():
        return scores

    def _extract(analysis: dict):
        house_id = str(analysis.get('house_id', ''))
        if not house_id:
            return None
        quality = analysis.get('data_quality', {})
        quality_label = quality.get('quality_label')
        quality_score = quality.get('quality_score')

        if quality_label and quality_label.startswith('faulty'):
            qs = quality_label
        elif quality_score is not None:
            qs = quality_score
        else:
            return None

        return house_id, {
            'quality_score': qs,
            'nan_continuity': quality.get('nan_continuity_label', 'unknown'),
            'max_nan_pct': quality.get('max_phase_nan_pct', 0),
        }

    if house_analysis_path.is_dir():
        per_house_dir = house_analysis_path / "per_house"
        if not per_house_dir.exists():
            per_house_dir = house_analysis_path

        for json_file in per_house_dir.glob("analysis_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    analysis = _json.load(f)
                result = _extract(analysis)
                if result:
                    scores[result[0]] = result[1]
            except Exception:
                pass
        return scores

    try:
        with open(house_analysis_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)

        analyses = data if isinstance(data, list) else data.get('analyses', [])
        for analysis in analyses:
            result = _extract(analysis)
            if result:
                scores[result[0]] = result[1]
    except Exception:
        pass

    return scores


def _assign_tier(pre_quality) -> str:
    """Assign quality tier based on pre-analysis quality score."""
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        return pre_quality
    elif pre_quality is None:
        return 'unknown'
    elif pre_quality >= 90:
        return 'excellent'
    elif pre_quality >= 75:
        return 'good'
    elif pre_quality >= 50:
        return 'fair'
    else:
        return 'poor'


def _format_pre_quality(pre_quality) -> str:
    """Format pre-quality score as colored HTML."""
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        _faulty_labels = {
            'faulty_dead_phase': ('Dead Phase', 'Phase with <2% of sisters avg'),
            'faulty_high_nan': ('High NaN', 'Phase with >=10% NaN values'),
            'faulty_both': ('Both', 'Dead phase + high NaN on other phases'),
        }
        _fl, _ft = _faulty_labels.get(pre_quality, ('Faulty', ''))
        return f'<span style="color:#6f42c1;font-weight:bold;" title="{_ft}">{_fl}</span>'
    elif pre_quality is None:
        return '<span style="color:#999;">-</span>'
    else:
        if pre_quality >= 90:
            color = '#28a745'
        elif pre_quality >= 75:
            color = '#007bff'
        elif pre_quality >= 50:
            color = '#ffc107'
        else:
            color = '#dc3545'
        return f'<span style="color:{color};font-weight:bold;">{pre_quality:.0f}</span>'


def _build_quality_dist_bar(tier_counts: dict, n_houses: int) -> str:
    """Build quality distribution bar HTML — shared visual pattern across all reports."""
    tier_config = [
        ('excellent', 'Excellent', '#28a745'),
        ('good', 'Good', '#007bff'),
        ('fair', 'Fair', '#ffc107'),
        ('poor', 'Poor', '#dc3545'),
        ('faulty_dead_phase', 'Faulty (Dead)', '#5a3d7a'),
        ('faulty_high_nan', 'Faulty (NaN)', '#6f42c1'),
        ('faulty_both', 'Faulty (Both)', '#4a0e6b'),
        ('unknown', 'Unknown', '#6c757d'),
    ]

    segments = ''
    legend_items = ''
    for key, label, color in tier_config:
        count = tier_counts.get(key, 0)
        if count == 0:
            continue
        pct = count / n_houses * 100 if n_houses > 0 else 0
        segments += (f'<div style="width:{pct:.1f}%;background:{color};height:100%;'
                     f'display:inline-block;" title="{label}: {count} ({pct:.0f}%)"></div>')
        legend_items += (f'<span style="display:inline-flex;align-items:center;gap:4px;'
                         f'margin-right:12px;font-size:0.82em;">'
                         f'<span style="width:10px;height:10px;border-radius:50%;'
                         f'background:{color};display:inline-block;"></span>'
                         f'{label}: {count} ({pct:.0f}%)</span>')

    return f'''
    <div style="margin:18px 0;">
        <div style="font-size:0.82em;font-weight:600;color:#555;margin-bottom:6px;">
            Input Quality Distribution</div>
        <div style="width:100%;height:18px;border-radius:9px;overflow:hidden;
                    background:#e9ecef;font-size:0;line-height:0;">{segments}</div>
        <div style="margin-top:6px;line-height:1.8;">{legend_items}</div>
    </div>'''


def _load_summarized_power(experiment_dir: Path, house_id: str):
    """Load summarized power: true original from first run + final remaining from last run.

    Each pipeline iteration stores original_wN (input to THAT iteration) and
    remaining_wN (output). So run_0 has the true raw original, and the last run
    has the final remaining after all device extractions. Loading only the last
    run would show original ≈ remaining (segregated ≈ 0) because most devices
    were already extracted in earlier iterations.

    Returns a DataFrame with columns: timestamp, original_wN, remaining_wN
    where original comes from run_0 and remaining from the last run.
    """
    import pandas as pd

    run_dirs = sorted(
        [d for d in experiment_dir.glob("run_*") if d.is_dir()],
        key=lambda d: d.name,
    )
    if not run_dirs:
        return None

    first_run = run_dirs[0]
    last_run = run_dirs[-1]

    # Load first run (true original power)
    first_df = _load_run_summarized_files(first_run, house_id)
    if first_df is None:
        return None

    # Single run — use as-is
    if first_run == last_run:
        return first_df

    # Load last run (final remaining power)
    last_df = _load_run_summarized_files(last_run, house_id)
    if last_df is None:
        return first_df

    # Merge: original_wN from first run + remaining_wN from last run
    phases = ['w1', 'w2', 'w3']
    orig_cols = ['timestamp'] + [f'original_{p}' for p in phases
                                  if f'original_{p}' in first_df.columns]
    remain_cols = ['timestamp'] + [f'remaining_{p}' for p in phases
                                    if f'remaining_{p}' in last_df.columns]

    merged = first_df[orig_cols].merge(last_df[remain_cols], on='timestamp', how='inner')

    if merged.empty:
        logger.warning("Merge produced empty result, falling back to first run data")
        return first_df

    logger.info(
        f"Loaded summarized power: {len(merged)} rows "
        f"(original from {first_run.name}, remaining from {last_run.name})"
    )
    return merged


def _load_run_summarized_files(run_dir: Path, house_id: str):
    """Load and concatenate summarized pkl files from a specific run directory."""
    import pandas as pd

    summarized_dir = run_dir / f"house_{house_id}" / "summarized"
    if not summarized_dir.exists():
        return None

    pkl_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not pkl_files:
        return None

    dfs = []
    for pkl_file in pkl_files:
        try:
            df = pd.read_pickle(pkl_file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {pkl_file}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_sessions(experiment_dir: Path, house_id: str) -> Dict[str, Any]:
    """Load device_sessions JSON for a house."""
    candidates = [
        experiment_dir / "device_sessions" / f"device_sessions_{house_id}.json",
        experiment_dir / f"device_sessions_{house_id}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sessions JSON from {path}: {e}")
    return {}


def _load_spike_intervals(experiment_dir: Path, house_id: str) -> dict:
    """Load filtered spike matches (< 3 min) as time intervals per phase.

    Returns dict: {phase: [(start_iso, end_iso), ...]}
    """
    import pandas as pd

    MIN_DURATION = 3  # minutes — matches identification config
    run_dirs = sorted(
        [d for d in experiment_dir.glob("run_*") if d.is_dir()],
        key=lambda d: d.name,
    )
    all_spikes = []
    seen = set()  # deduplicate across runs by (phase, on_start)

    for run_dir in run_dirs:
        matches_dir = run_dir / f"house_{house_id}" / "matches"
        if not matches_dir.exists():
            continue
        for pkl_file in sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if df.empty or 'duration' not in df.columns:
                    continue
                spikes = df[df['duration'] < MIN_DURATION].copy()
                if spikes.empty:
                    continue
                # Deduplicate: same physical event may appear in multiple runs
                spikes['_key'] = spikes['phase'] + '_' + spikes['on_start'].astype(str)
                new_spikes = spikes[~spikes['_key'].isin(seen)]
                if not new_spikes.empty:
                    seen.update(new_spikes['_key'].tolist())
                    all_spikes.append(new_spikes.drop(columns=['_key']))
            except Exception:
                continue

    if not all_spikes:
        return {}

    combined = pd.concat(all_spikes, ignore_index=True)
    result = {}
    for phase in ['w1', 'w2', 'w3']:
        phase_spikes = combined[combined['phase'] == phase]
        if phase_spikes.empty:
            continue
        starts = pd.to_datetime(phase_spikes['on_start'])
        # Use off_end if available, fallback to off_start, then on_start
        if 'off_end' in phase_spikes.columns:
            ends = pd.to_datetime(phase_spikes['off_end'].fillna(
                phase_spikes.get('off_start', phase_spikes['on_start'])))
        elif 'off_start' in phase_spikes.columns:
            ends = pd.to_datetime(phase_spikes['off_start'])
        else:
            ends = starts
        intervals = list(zip(
            starts.apply(lambda x: x.isoformat()),
            ends.apply(lambda x: x.isoformat()),
        ))
        result[phase] = sorted(intervals)

    logger.info(f"Loaded {sum(len(v) for v in result.values())} spike intervals for house {house_id}")
    return result


def _load_all_match_intervals(experiment_dir: Path, house_id: str) -> dict:
    """Load all matches from all iterations as intervals per phase.

    Returns dict: {phase: [(on_start_iso, off_end_iso, magnitude, duration,
                            on_end_iso, off_start_iso), ...]}
    Used by chart visualization to show individual match shapes.
    on_end and off_start enable trapezoidal rendering (ramp up → stable → ramp down)
    as fallback when per-minute power_profile is not available.
    """
    import pandas as pd

    run_dirs = sorted(
        [d for d in experiment_dir.glob("run_*") if d.is_dir()],
        key=lambda d: d.name,
    )
    all_dfs = []
    for run_dir in run_dirs:
        matches_dir = run_dir / f"house_{house_id}" / "matches"
        if not matches_dir.exists():
            continue
        for pkl_file in sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if not df.empty:
                    cols = ['phase', 'on_start', 'on_end', 'off_start', 'off_end',
                            'on_magnitude', 'duration']
                    available = [c for c in cols if c in df.columns]
                    all_dfs.append(df[available].copy())
            except Exception:
                continue

    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate by (phase, on_start)
    combined['_key'] = combined['phase'] + '_' + combined['on_start'].astype(str)
    combined = combined.drop_duplicates(subset='_key', keep='first').drop(columns=['_key'])
    # Fill missing off_end
    if 'off_end' in combined.columns:
        combined['off_end'] = combined['off_end'].fillna(
            combined.get('off_start', combined['on_start']))
    else:
        combined['off_end'] = combined.get('off_start', combined['on_start'])
    # Fill missing on_end / off_start with on_start / off_end respectively
    if 'on_end' not in combined.columns:
        combined['on_end'] = combined['on_start']
    else:
        combined['on_end'] = combined['on_end'].fillna(combined['on_start'])
    if 'off_start' not in combined.columns:
        combined['off_start'] = combined['off_end']
    else:
        combined['off_start'] = combined['off_start'].fillna(combined['off_end'])

    result = {}
    for phase in ['w1', 'w2', 'w3']:
        pm = combined[combined['phase'] == phase]
        if pm.empty:
            continue
        starts = pd.to_datetime(pm['on_start']).apply(lambda x: x.isoformat())
        ends = pd.to_datetime(pm['off_end']).apply(lambda x: x.isoformat())
        on_ends = pd.to_datetime(pm['on_end']).apply(lambda x: x.isoformat())
        off_starts = pd.to_datetime(pm['off_start']).apply(lambda x: x.isoformat())
        mags = pm['on_magnitude'].abs().round().astype(int)
        durs = pm['duration'].fillna(0).round(1)
        intervals = sorted(zip(starts, ends, mags, durs, on_ends, off_starts))
        result[phase] = intervals

    logger.info(f"Loaded {sum(len(v) for v in result.values())} match intervals for house {house_id}")
    return result


def _load_activations(experiment_dir: Path, house_id: str) -> list:
    """Load device_activations JSON (flat format) for backward-compatible detail tables."""
    candidates = [
        experiment_dir / "device_activations" / f"device_activations_{house_id}.json",
        experiment_dir / f"device_activations_{house_id}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return data.get('activations', [])
            except Exception as e:
                logger.warning(f"Failed to load activations JSON from {path}: {e}")
    return []


# ---------------------------------------------------------------------------
# Per-house report
# ---------------------------------------------------------------------------

def generate_identification_report(
    experiment_dir: str,
    house_id: str,
    output_path: Optional[str] = None,
    skip_activations_detail: bool = False,
    show_timing: bool = False,
) -> dict:
    """
    Generate device identification HTML report for a single house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        output_path: Where to save the HTML file (optional)
        skip_activations_detail: If True, omit the Device Activations Detail section
        show_timing: If True, print per-step timing to console

    Returns:
        dict with keys: 'path' (str), 'quality' (dict or None), 'confidence' (dict or None)
    """
    import time as _time

    experiment_dir = Path(experiment_dir)
    prefix = f"  [house {house_id}]"
    total_t0 = _time.time()

    def _step(label):
        """Helper: print timing for previous step, start new one."""
        now = _time.time()
        if _step._prev is not None:
            elapsed = now - _step._start
            if show_timing:
                print(f"{prefix} {_step._prev:<40s} {elapsed:.1f}s", flush=True)
        _step._prev = label
        _step._start = now
    _step._prev = None
    _step._start = total_t0

    # Load data
    _step("Loading sessions...")
    sessions_data = _load_sessions(experiment_dir, house_id)
    sessions = sessions_data.get('sessions', [])
    threshold_schedule = sessions_data.get('threshold_schedule', [])

    # Generate chart sections
    _step("Creating overview charts...")
    spike_filter = sessions_data.get('spike_filter', {})
    spike_html = create_spike_analysis(spike_filter)
    overview_html = create_session_overview(sessions)
    confidence_html = create_confidence_overview(sessions)
    boiler_html = create_boiler_analysis(sessions)
    ac_html = create_ac_analysis(sessions)
    heatmap_html = create_temporal_heatmap(sessions)
    unclassified_html = create_unclassified_analysis(sessions)

    # Classification quality metrics (reuse existing M2 metrics)
    _step("Classification quality metrics...")
    quality_html = ''
    quality = None
    confidence = None
    try:
        quality = calculate_classification_quality(experiment_dir, house_id)
        confidence = calculate_confidence_scores(experiment_dir, house_id)
        quality_html = create_quality_section(quality, confidence)
    except Exception as e:
        logger.warning(f"Classification quality failed for house {house_id}: {e}")

    # Load summarized power data for expandable charts in activations table
    _step("Loading summarized power...")
    summarized_data = None
    try:
        summarized_data = _load_summarized_power(experiment_dir, house_id)
    except Exception as e:
        logger.warning(f"Failed to load summarized power for house {house_id}: {e}")

    # Load all match intervals for chart visualization (individual match rectangles)
    _step("Loading match intervals...")
    all_match_intervals = {}
    try:
        all_match_intervals = _load_all_match_intervals(experiment_dir, house_id)
    except Exception as e:
        logger.warning(f"Failed to load match intervals for house {house_id}: {e}")

    # Device activations detail (session-level) with expandable power charts
    activations_detail_html = ''
    if not skip_activations_detail:
        _step("Device activations detail...")
        activations_detail_html = create_device_activations_detail(
            sessions, house_id=house_id, summarized_data=summarized_data,
            all_match_intervals=all_match_intervals,
        )
    else:
        _step("Skipping activations detail...")

    # Build HTML
    _step("Building HTML...")
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    summary = sessions_data.get('summary', {})

    html = _build_house_html(
        house_id=house_id,
        generated_at=generated_at,
        threshold_schedule=threshold_schedule,
        spike_html=spike_html,
        overview_html=overview_html,
        confidence_html=confidence_html,
        boiler_html=boiler_html,
        ac_html=ac_html,
        heatmap_html=heatmap_html,
        quality_html=quality_html,
        unclassified_html=unclassified_html,
        activations_detail_html=activations_detail_html,
        summary=summary,
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / f"identification_report_{house_id}.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # Final timing
    _step("Done")
    total_elapsed = _time.time() - total_t0
    if show_timing:
        print(f"{prefix} {'TOTAL':<40s} {total_elapsed:.1f}s", flush=True)

    logger.info(f"Identification report saved to {output_path}")
    return {'path': output_path, 'quality': quality, 'confidence': confidence}


def _build_house_html(
    house_id: str,
    generated_at: str,
    threshold_schedule: list,
    spike_html: str,
    overview_html: str,
    confidence_html: str,
    boiler_html: str,
    ac_html: str,
    heatmap_html: str,
    quality_html: str,
    unclassified_html: str,
    activations_detail_html: str,
    summary: Dict[str, Any],
) -> str:
    """Build complete HTML document for a single house identification report."""
    th_str = ' \u2192 '.join(f'{t}W' for t in threshold_schedule) if threshold_schedule else 'N/A'
    total_sessions = summary.get('total_sessions', 0)
    by_type = summary.get('by_device_type', {})

    # Type summary for header
    type_parts = []
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'unknown']:
        info = by_type.get(dtype, {})
        count = info.get('count', 0)
        if count > 0:
            type_parts.append(f'{dtype}: {count}')
    type_str = ', '.join(type_parts) if type_parts else 'No sessions'

    # Activations detail section
    activations_section_html = ''
    if activations_detail_html:
        activations_section_html = f'''
        <section>
            <h2>Device Activations Detail</h2>
            <p style="color: #666; margin-bottom: 10px; font-size: 0.85em;">
                Individual ON&rarr;OFF activations grouped by device type.
                Click column headers to sort. Use "Copy Dates" for external tools.
            </p>
            {activations_detail_html}
        </section>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Identification Report - House {house_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a365d 0%, #2a4365 50%, #2c5282 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 5px; }}
        header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .info-bar {{ display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap; }}
        .info-item {{ font-size: 0.9em; opacity: 0.85; }}
        .info-item strong {{ opacity: 1; }}
        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{
            color: #2a4365;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; }}
        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .info-bar {{ flex-direction: column; gap: 5px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Device Identification Report</h1>
            <div class="subtitle">House {house_id} &mdash; Module 2 Analysis</div>
            <div class="info-bar">
                <div class="info-item"><strong>Generated:</strong> {generated_at}</div>
                <div class="info-item"><strong>Thresholds:</strong> {th_str}</div>
                <div class="info-item"><strong>Sessions:</strong> {total_sessions} ({type_str})</div>
            </div>
        </header>

        <section>
            <h2>Transient Event Filter (Spikes)</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                This project works with 1-minute resolution data and uses a purely unsupervised,
                heuristic approach &mdash; no ML, no features beyond power magnitude. We focus on devices
                with consistent usage patterns (boiler, central AC, regular AC). Transient events
                under 3 minutes (microwave, oven, motor starts) cannot be reliably identified at this
                resolution, and no classification rule accepts them. Filtering is a statement of scope.
            </p>
            {spike_html}
        </section>

        <section>
            <h2>Session Overview</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Summary of all detected device sessions. Each session is a group of temporally close
                matched ON&rarr;OFF events on the same phase, classified by power, duration, and phase patterns.
            </p>
            {overview_html}
        </section>

        <section>
            <h2>Confidence Distribution</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Confidence scores (0&ndash;1) for classified sessions based on how well each session
                matches its device type criteria (duration, magnitude, isolation, phase consistency).
            </p>
            {confidence_html}
        </section>

        <section>
            <h2>Boiler Analysis</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Water heater detection analysis: daily usage patterns, monthly consistency,
                power magnitude stability, and dominant phase.
                Boiler criteria: &ge;15 min, &ge;1500W, single-phase, isolated.
            </p>
            {boiler_html}
        </section>

        <section>
            <h2>AC Analysis</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Air conditioning detection: Central AC (multi-phase synchronized) vs Regular AC
                (single-phase compressor cycling). Seasonal patterns and cycle characteristics.
            </p>
            {ac_html}
        </section>

        <section>
            <h2>Temporal Patterns</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                When are devices active? Heatmap shows session start times by hour and day of week.
                Expected patterns: boiler morning/evening, AC afternoon/night in summer.
            </p>
            {heatmap_html}
        </section>

        {quality_html}

        <section>
            <h2>Unclassified Sessions</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Sessions that did not match any device classification rule.
                These may represent small appliances, partial detections, or device types
                not covered by the current heuristic rules.
            </p>
            {unclassified_html}
        </section>

        {activations_section_html}

        <footer>
            ElectricPatterns &mdash; Module 2: Device Identification Report
        </footer>
    </div>

    <script>
    var deviceSortState = {{}};
    function sortDeviceTable(tableId, colIdx, type) {{
        var table = document.getElementById(tableId);
        if (!table) return;
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var key = tableId + '-' + colIdx;
        var asc = deviceSortState[key] === undefined ? true : !deviceSortState[key];
        deviceSortState[key] = asc;
        rows.sort(function(a, b) {{
            var cellA = a.cells[colIdx], cellB = b.cells[colIdx];
            var vA, vB;
            if (type === 'num') {{
                vA = parseFloat(cellA.getAttribute('data-value') || cellA.textContent.replace(/[^0-9.-]/g, '')) || 0;
                vB = parseFloat(cellB.getAttribute('data-value') || cellB.textContent.replace(/[^0-9.-]/g, '')) || 0;
            }} else {{
                vA = (cellA.getAttribute('data-value') || cellA.textContent).trim();
                vB = (cellB.getAttribute('data-value') || cellB.textContent).trim();
                if (vA < vB) return asc ? -1 : 1;
                if (vA > vB) return asc ? 1 : -1;
                return 0;
            }}
            return asc ? (vA - vB) : (vB - vA);
        }});
        rows.forEach(function(row, i) {{
            row.cells[0].textContent = i + 1;
            tbody.appendChild(row);
        }});
    }}
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Aggregate report (cross-house)
# ---------------------------------------------------------------------------

def generate_identification_aggregate_report(
    experiment_dir: str,
    house_ids: List[str],
    output_path: Optional[str] = None,
    house_reports_subdir: Optional[str] = None,
    show_progress: bool = False,
    precomputed_metrics: Optional[Dict[str, dict]] = None,
    show_timing: bool = False,
    per_house_filename_pattern: Optional[str] = None,
    pre_analysis_scores: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate aggregate identification report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)
        house_reports_subdir: Subdirectory for per-house report links
        show_progress: Show tqdm progress bar
        precomputed_metrics: Optional dict {house_id: {'quality': ..., 'confidence': ...}}
                             from per-house phase — avoids recalculating
        show_timing: If True, print timing info

    Returns:
        Path to generated HTML file
    """
    import time as _time

    experiment_dir = Path(experiment_dir)
    precomputed_metrics = precomputed_metrics or {}

    all_quality = []
    all_confidence = []
    house_summaries = []

    houses_iter = house_ids
    if show_progress and _HAS_TQDM:
        houses_iter = _tqdm(house_ids, desc="Aggregate M2 metrics", unit="house")

    t0 = _time.time()
    reused = 0

    for house_id in houses_iter:
        sessions_data = _load_sessions(experiment_dir, house_id)
        sessions = sessions_data.get('sessions', [])
        summary = sessions_data.get('summary', {})

        # Spike filter info
        spike_filter = sessions_data.get('spike_filter', {})
        spike_count = spike_filter.get('spike_count', 0)

        # Report link (always generate, even for empty houses)
        report_link = None
        if house_reports_subdir:
            if per_house_filename_pattern:
                per_house_file = per_house_filename_pattern.replace('{house_id}', house_id)
            else:
                per_house_file = f'identification_report_{house_id}.html'
            report_link = f'{house_reports_subdir}/{per_house_file}'

        # Pre-analysis quality for this house
        house_pre = (pre_analysis_scores or {}).get(house_id, {})
        if isinstance(house_pre, dict):
            pre_quality = house_pre.get('quality_score')
        else:
            pre_quality = house_pre

        if not sessions:
            house_summaries.append({
                'house_id': house_id,
                'total_sessions': 0,
                'classified': 0,
                'classified_pct': 0,
                'avg_confidence': 0,
                'quality_score': None,
                'device_counts': {},
                'report_link': report_link,
                'days_span': 0,
                'sessions_per_day': 0,
                'spike_count': spike_count,
                'pre_quality': pre_quality,
            })
            continue

        # Collect per-house summary
        total = len(sessions)
        classified = sum(1 for s in sessions if s.get('device_type') not in ('unknown', 'unclassified'))
        conf_vals = [s.get('confidence', 0) for s in sessions if s.get('confidence')]
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0

        # Device counts
        device_counts = {}
        for s in sessions:
            dt = s.get('device_type', 'unknown')
            device_counts[dt] = device_counts.get(dt, 0) + 1

        # Days span and sessions/day from session timestamps
        session_dates = set()
        min_date = max_date = None
        for s in sessions:
            start = s.get('start', '')
            if start:
                try:
                    d = datetime.fromisoformat(str(start)).date()
                    session_dates.add(d)
                    if min_date is None or d < min_date:
                        min_date = d
                    if max_date is None or d > max_date:
                        max_date = d
                except (ValueError, TypeError):
                    pass

        days_span = (max_date - min_date).days + 1 if min_date and max_date else 0
        sessions_per_day = total / days_span if days_span > 0 else 0

        # Quality + confidence metrics — reuse from per-house phase if available
        cached = precomputed_metrics.get(house_id, {})
        quality = cached.get('quality')
        confidence = cached.get('confidence')

        if quality is None:
            try:
                quality = calculate_classification_quality(experiment_dir, house_id)
            except Exception:
                quality = None
        else:
            reused += 1

        if confidence is None:
            try:
                confidence = calculate_confidence_scores(experiment_dir, house_id)
            except Exception:
                confidence = None

        if quality:
            all_quality.append(quality)
        if confidence:
            all_confidence.append(confidence)

        quality_score = quality.get('overall_quality_score') if quality else None

        house_summaries.append({
            'house_id': house_id,
            'total_sessions': total,
            'classified': classified,
            'classified_pct': classified / total * 100 if total > 0 else 0,
            'avg_confidence': avg_conf,
            'quality_score': quality_score,
            'device_counts': device_counts,
            'report_link': report_link,
            'days_span': days_span,
            'sessions_per_day': sessions_per_day,
            'spike_count': spike_count,
            'pre_quality': pre_quality,
        })

    if show_timing:
        metrics_time = _time.time() - t0
        cache_msg = f", {reused} reused from cache" if reused else ""
        print(f"  Aggregate: collected metrics for {len(house_summaries)} houses "
              f"({metrics_time:.1f}s{cache_msg})", flush=True)

    # Population statistics
    t1 = _time.time()
    population_stats = {}
    if all_quality and all_confidence:
        try:
            population_stats = compute_population_statistics(all_quality, all_confidence)
        except Exception as e:
            logger.warning(f"Population statistics failed: {e}")
    if show_timing:
        print(f"  Aggregate: population statistics ({_time.time() - t1:.1f}s)", flush=True)

    # Build HTML
    t2 = _time.time()
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = _build_aggregate_html(
        generated_at=generated_at,
        house_summaries=house_summaries,
        population_stats=population_stats,
        experiment_dir=str(experiment_dir),
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / "identification_report_aggregate.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    if show_timing:
        print(f"  Aggregate: built HTML + saved ({_time.time() - t2:.1f}s)", flush=True)

    logger.info(f"Aggregate identification report saved to {output_path}")
    return output_path


def _build_aggregate_html(
    generated_at: str,
    house_summaries: List[Dict],
    population_stats: Dict[str, Any],
    experiment_dir: str,
) -> str:
    """Build complete aggregate HTML document."""
    n = len(house_summaries)
    if n == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir)

    # Aggregate stats
    avg_classified = sum(h['classified_pct'] for h in house_summaries) / n
    avg_conf = sum(h['avg_confidence'] for h in house_summaries) / n
    quality_scores = [h['quality_score'] for h in house_summaries if h['quality_score'] is not None]
    median_quality = sorted(quality_scores)[len(quality_scores) // 2] if quality_scores else 0

    # Total data days
    total_days = sum(h.get('days_span', 0) for h in house_summaries)

    # Device detection rates
    has_boiler = sum(1 for h in house_summaries if h['device_counts'].get('boiler', 0) > 0)
    has_central = sum(1 for h in house_summaries if h['device_counts'].get('central_ac', 0) > 0)
    has_regular = sum(1 for h in house_summaries if h['device_counts'].get('regular_ac', 0) > 0)

    # Input quality tier counts
    tier_counts = {}
    for h in house_summaries:
        tier = _assign_tier(h.get('pre_quality'))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    quality_dist_bar = _build_quality_dist_bar(tier_counts, n)

    # Population stats section
    pop_html = _build_population_section(population_stats) if population_stats else ''

    # Per-house table rows
    _td = 'padding:8px 10px;border-bottom:1px solid #eee;'
    table_rows = ''
    for h in sorted(house_summaries, key=lambda x: x['house_id']):
        hid = h['house_id']
        link = f'<a href="{h["report_link"]}" style="color: #667eea; text-decoration: none;">{hid}</a>' if h.get('report_link') else hid

        # Pre-quality
        pq_html = _format_pre_quality(h.get('pre_quality'))
        tier = _assign_tier(h.get('pre_quality'))

        # Device type badges
        badges = ''
        for dt, color in [('boiler', '#007bff'), ('central_ac', '#dc3545'), ('regular_ac', '#e67e22')]:
            cnt = h['device_counts'].get(dt, 0)
            if cnt > 0:
                badges += f'<span style="background:{color};color:white;padding:1px 6px;border-radius:8px;font-size:0.75em;margin-right:3px;">{dt.split("_")[0]} {cnt}</span>'

        q = h['quality_score']
        q_str = f'{q:.2f}' if q is not None else '-'
        q_val = q if q is not None else 0
        q_color = '#28a745' if q and q >= 0.8 else '#eab308' if q and q >= 0.4 else '#e67e22' if q else '#aaa'
        c_color = '#28a745' if h['avg_confidence'] >= 0.8 else '#eab308' if h['avg_confidence'] >= 0.4 else '#e67e22'

        days = h.get('days_span', 0)
        spd = h.get('sessions_per_day', 0)
        spikes = h.get('spike_count', 0)

        table_rows += f'''
        <tr data-tier="{tier}">
            <td style="{_td}" data-value="{hid}">{link}</td>
            <td style="{_td}text-align:center;" data-value="{days}">{days}</td>
            <td style="{_td}text-align:center;">{pq_html}</td>
            <td style="{_td}text-align:center;" data-value="{h['total_sessions']}">{h['total_sessions']}</td>
            <td style="{_td}text-align:center;" data-value="{spd:.2f}">{spd:.1f}</td>
            <td style="{_td}text-align:center;" data-value="{spikes}">{spikes}</td>
            <td style="{_td}text-align:center;" data-value="{h['classified_pct']:.1f}">{h['classified_pct']:.0f}%</td>
            <td style="{_td}text-align:center;color:{c_color};font-weight:600;" data-value="{h['avg_confidence']:.3f}">{h['avg_confidence']:.2f}</td>
            <td style="{_td}text-align:center;color:{q_color};font-weight:600;" data-value="{q_val:.3f}">{q_str}</td>
            <td style="{_td}">{badges}</td>
        </tr>'''

    # Tier filter bar
    _tier_styles = {
        'excellent': ('Excellent', '#d4edda', '#155724'),
        'good': ('Good', '#cce5ff', '#004085'),
        'fair': ('Fair', '#fff3cd', '#856404'),
        'poor': ('Poor', '#fde2d4', '#813e1a'),
        'faulty_dead_phase': ('Dead Phase', '#d4c5e2', '#5a3d7a'),
        'faulty_high_nan': ('High NaN', '#e2d5f0', '#6f42c1'),
        'faulty_both': ('Faulty Both', '#c9a3d4', '#4a0e6b'),
        'unknown': ('Unknown', '#e9ecef', '#495057'),
    }
    filter_checkboxes = ''
    for tier_key, (label, bg, fg) in _tier_styles.items():
        cnt = tier_counts.get(tier_key, 0)
        if cnt == 0:
            continue
        filter_checkboxes += (
            f'<label style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
            f'border-radius:4px;font-size:0.85em;background:{bg};color:{fg};cursor:pointer;">'
            f'<input type="checkbox" data-filter-tier="{tier_key}" checked '
            f'onchange="updateIdFilter()"> {label} ({cnt})</label> '
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Identification - Aggregate Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa; color: #333; line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a365d 0%, #2a4365 50%, #2c5282 100%);
            color: white; padding: 30px; margin-bottom: 30px; border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 5px; }}
        section {{
            background: white; border-radius: 10px; padding: 25px;
            margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{ color: #2a4365; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }}
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 25px; border-radius: 10px; text-align: center;
        }}
        .summary-number {{ font-size: 2.2em; font-weight: bold; color: #2d3748; }}
        .summary-label {{ color: #666; margin-top: 5px; font-size: 0.9em; }}
        .filter-bar {{
            display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
            margin-bottom: 12px; padding: 10px 15px; background: #f8f9fa; border-radius: 8px;
        }}
        .filter-bar label {{ font-weight: 600; color: #555; margin-right: 5px; }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Device Identification &mdash; Aggregate Report</h1>
            <div style="opacity:0.9;">Generated: {generated_at}</div>
        </header>

        <section>
            <h2>Summary</h2>
            <!-- Row 1: Houses + Days -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px;">
                <div class="summary-card">
                    <div class="summary-number">{n}</div>
                    <div class="summary-label">Houses Analyzed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number">{total_days:,}</div>
                    <div class="summary-label">Total Days of Data</div>
                </div>
            </div>
            <!-- Row 2: Quality distribution -->
            {quality_dist_bar}
            <!-- Row 3: Report-specific metrics -->
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px;margin-bottom:20px;">
                <div style="background:#d4edda;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#28a745;">{avg_classified:.0f}%</div>
                    <div style="font-size:0.85em;color:#666;">Avg Classified</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#667eea;">{avg_conf:.2f}</div>
                    <div style="font-size:0.85em;color:#666;">Avg Confidence</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#667eea;">{median_quality:.2f}</div>
                    <div style="font-size:0.85em;color:#666;">Median Quality</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div style="background:#f0f4ff;border:1px solid #c3d4ff;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#007bff;">{has_boiler}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Boiler</div>
                </div>
                <div style="background:#fff5f5;border:1px solid #fed7d7;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#dc3545;">{has_central}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Central AC</div>
                </div>
                <div style="background:#fff8f0;border:1px solid #feebc8;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#e67e22;">{has_regular}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Regular AC</div>
                </div>
            </div>
        </section>

        {pop_html}

        <section>
            <h2>Per-House Results</h2>
            <div class="filter-bar">
                <span style="font-weight:600;color:#555;margin-right:5px;">Quality Tier:</span>
                {filter_checkboxes}
                <span id="id-filter-status" style="font-size:0.85em;color:#888;margin-left:auto;"></span>
            </div>
            <div style="font-size:0.82em;color:#666;margin-bottom:10px;line-height:1.7;">
                <strong>Column descriptions:</strong>
                <strong>Days</strong> = calendar days from first to last session |
                <strong>Pre-Quality</strong> = input data quality from house pre-analysis |
                <strong>Sessions</strong> = total device sessions found |
                <strong>Sess/Day</strong> = average sessions per day (higher = more device activity detected) |
                <strong>Spikes</strong> = transient events filtered out (&lt;3 min) |
                <strong>Classified</strong> = % of sessions assigned to a device type (boiler/AC) |
                <strong>Confidence</strong> = avg classification confidence (0&ndash;1, how well each session matches its device criteria) |
                <strong>Quality</strong> = internal consistency score (temporal, magnitude, duration, seasonal checks)
            </div>
            <div style="overflow-x:auto;">
            <table id="agg-table" style="width:100%;border-collapse:collapse;font-size:0.88em;">
                <thead>
                    <tr style="background:#2d3748;color:white;">
                        <th style="padding:8px 10px;text-align:left;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(0,'str')" title="House identifier">House &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(1,'num')" title="Calendar days from first to last session">Days &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(2,'num')" title="Input data quality score from house pre-analysis">Pre-Quality &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(3,'num')" title="Total device sessions found">Sessions &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(4,'num')" title="Average sessions per day">Sess/Day &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(5,'num')" title="Transient events filtered (<3 min)">Spikes &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(6,'num')" title="% of sessions assigned to a device type">Classified &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(7,'num')" title="Average classification confidence (0-1)">Confidence &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(8,'num')" title="Internal consistency quality score (0-1)">Quality &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Devices</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            </div>
        </section>

        <footer>
            ElectricPatterns &mdash; Module 2: Device Identification Aggregate Report
        </footer>
    </div>

    <script>
    var aggSortState = {{}};
    function sortAggTable(colIdx, type) {{
        var table = document.getElementById('agg-table');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var key = 'agg-' + colIdx;
        var asc = aggSortState[key] === undefined ? true : !aggSortState[key];
        aggSortState[key] = asc;
        rows.sort(function(a, b) {{
            var cellA = a.cells[colIdx], cellB = b.cells[colIdx];
            var vA, vB;
            if (type === 'num') {{
                vA = parseFloat(cellA.getAttribute('data-value') || cellA.textContent.replace(/[^0-9.-]/g, '')) || 0;
                vB = parseFloat(cellB.getAttribute('data-value') || cellB.textContent.replace(/[^0-9.-]/g, '')) || 0;
            }} else {{
                vA = (cellA.getAttribute('data-value') || cellA.textContent).trim();
                vB = (cellB.getAttribute('data-value') || cellB.textContent).trim();
                if (vA < vB) return asc ? -1 : 1;
                if (vA > vB) return asc ? 1 : -1;
                return 0;
            }}
            return asc ? (vA - vB) : (vB - vA);
        }});
        rows.forEach(function(row) {{ tbody.appendChild(row); }});
        var ths = table.querySelectorAll('thead th');
        ths.forEach(function(th, i) {{
            th.style.fontWeight = (i === colIdx) ? '900' : 'normal';
        }});
    }}

    function updateIdFilter() {{
        var checkedTiers = [];
        document.querySelectorAll('[data-filter-tier]').forEach(function(cb) {{
            if (cb.checked) checkedTiers.push(cb.getAttribute('data-filter-tier'));
        }});
        var rows = document.querySelectorAll('#agg-table tbody tr');
        var shown = 0;
        rows.forEach(function(row) {{
            var tier = row.getAttribute('data-tier') || 'unknown';
            if (checkedTiers.indexOf(tier) !== -1) {{
                row.style.display = '';
                shown++;
            }} else {{
                row.style.display = 'none';
            }}
        }});
        var status = document.getElementById('id-filter-status');
        if (status) status.textContent = 'Showing ' + shown + ' / ' + rows.length + ' houses';
    }}
    updateIdFilter();
    </script>
</body>
</html>"""


def _build_population_section(population_stats: Dict[str, Any]) -> str:
    """Build HTML section for population-level statistics."""
    if not population_stats or population_stats.get('houses_analyzed', 0) == 0:
        return ''

    n = population_stats['houses_analyzed']
    outliers = population_stats.get('outlier_houses', [])
    per_device = population_stats.get('per_device_type', {})

    device_cards = ''
    for dtype in ['boiler', 'central_ac', 'regular_ac']:
        if dtype not in per_device:
            continue
        d = per_device[dtype]
        count_dist = d.get('count_per_month', {})
        mag_dist = d.get('mean_magnitude', {})
        dur_dist = d.get('median_duration', {})
        houses_with = d.get("houses_with_device", 0)
        pct = (houses_with / n * 100) if n > 0 else 0
        device_cards += f'''
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">
            <div style="font-weight:600;color:#2a4365;margin-bottom:6px;">{dtype.replace("_", " ").title()}</div>
            <div style="font-size:0.85em;color:#555;">
                Houses: <strong>{houses_with}/{n}</strong> ({pct:.0f}%)<br>
                Median magnitude: {mag_dist.get("median", 0):.0f}W<br>
                Median duration: {dur_dist.get("median", 0):.0f} min<br>
                Months active/house: median {count_dist.get("median", 0):.0f}
            </div>
        </div>'''

    outlier_html = ''
    if outliers:
        outlier_list = ', '.join(str(o.get('house_id', '')) for o in outliers[:10])
        outlier_html = f'''
        <div style="margin-top:12px;padding:10px;background:#fff3cd;border:1px solid #ffc107;border-radius:6px;font-size:0.85em;">
            <strong>Outlier houses ({len(outliers)}):</strong> {outlier_list}
        </div>'''

    return f'''
    <section>
        <h2>Population Statistics</h2>
        <p style="color:#666;margin-bottom:12px;font-size:0.85em;">
            Cross-house analysis of device identification patterns.
            Z-scores flag houses with unusual device characteristics.
        </p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            {device_cards}
        </div>
        {outlier_html}
    </section>'''


def _build_empty_aggregate_html(generated_at: str, experiment_dir: str) -> str:
    """Fallback HTML when no valid data found."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Device Identification - Aggregate Report</title>
</head>
<body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px;">
    <h1>Device Identification - Aggregate Report</h1>
    <p>Generated: {generated_at}</p>
    <p style="color: #888;">No device session data found in: {experiment_dir}</p>
    <p>Ensure the pipeline was run with device identification enabled.</p>
</body>
</html>"""
