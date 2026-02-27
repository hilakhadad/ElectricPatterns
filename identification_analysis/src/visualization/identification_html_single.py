"""
Per-house identification report builders.

Extracted from identification_html_report.py.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import (
    assign_tier as _assign_tier,
    format_pre_quality as _format_pre_quality,
    build_glossary_section as _build_glossary_section,
    build_about_section as _build_about_section,
    build_upstream_metric_banner as _build_upstream_metric_banner,
)

from metrics.classification_quality import calculate_classification_quality
from metrics.confidence_scoring import calculate_confidence_scores

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

    # Compute data days and experiment name for header
    data_days = None
    if summarized_data is not None and 'timestamp' in summarized_data.columns:
        try:
            import pandas as pd
            ts = pd.to_datetime(summarized_data['timestamp'])
            data_days = (ts.max() - ts.min()).days
        except Exception:
            pass
    import re as _re
    _m = _re.match(r'^(.+?)_\d{8}_\d{6}$', experiment_dir.name)
    experiment_name = _m.group(1) if _m else (experiment_dir.name if experiment_dir.name.startswith('exp') else '')

    # Compute segregation effectiveness from summarized power (M1 upstream metric)
    segregation_pct = None
    if summarized_data is not None:
        try:
            phases = ['w1', 'w2', 'w3']
            total_original = sum(
                summarized_data[f'original_{p}'].sum()
                for p in phases if f'original_{p}' in summarized_data.columns
            )
            total_remaining = sum(
                summarized_data[f'remaining_{p}'].sum()
                for p in phases if f'remaining_{p}' in summarized_data.columns
            )
            if total_original > 0:
                segregation_pct = round(
                    max(0, (total_original - total_remaining) / total_original * 100), 1
                )
        except Exception as e:
            logger.debug(f"Could not compute segregation_pct: {e}")

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
        segregation_pct=segregation_pct,
        experiment_name=experiment_name or '',
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
    segregation_pct: Optional[float] = None,
    experiment_name: str = '',
) -> str:
    """Build complete HTML document for a single house identification report."""
    th_str = ' -> '.join(f'{t}W' for t in threshold_schedule) if threshold_schedule else 'N/A'
    total_sessions = summary.get('total_sessions', 0)
    by_type = summary.get('by_device_type', {})

    # Type summary for header
    type_parts = []
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown']:
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

    about_html = _build_about_section('identification')
    glossary_html = _build_glossary_section()

    # Upstream metric banner: segregation effectiveness from M1
    upstream_banner_html = ''
    if segregation_pct is not None:
        upstream_banner_html = _build_upstream_metric_banner(
            label='Segregation Effectiveness (from M1)',
            value=segregation_pct,
            suffix='%',
            color='#7B9BC4',
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Identification Report{' (' + experiment_name + ')' if experiment_name else ''} - House {house_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #FAFBFF;
            color: #3D3D50;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
            padding: 40px 30px;
            margin-bottom: 30px;
            border-radius: 16px;
        }}
        header h1 {{ font-size: 2.2em; margin-bottom: 5px; letter-spacing: -0.3px; font-weight: 700; }}
        header .subtitle {{ opacity: 0.92; font-size: 1.1em; }}
        .info-bar {{ display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap; }}
        .info-item {{ font-size: 0.9em; opacity: 0.85; }}
        .info-item strong {{ opacity: 1; }}
        section {{
            background: #FFFFFF;
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 22px;
            box-shadow: 0 2px 12px rgba(120,100,160,0.07);
            border: 1px solid #E8E4F0;
        }}
        section h2 {{
            color: #3D3D50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #E8E4F0;
            font-size: 1.35em;
        }}
        footer {{ text-align: center; padding: 20px; color: #7D7D92; font-size: 0.9em; }}
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
            <h1>Identification Report - House {house_id}</h1>
            <div class="subtitle">{'<strong>' + experiment_name + '</strong> | ' if experiment_name else ''}Module 2 Analysis</div>
            <div class="info-bar">
                <div class="info-item"><strong>Generated:</strong> {generated_at}</div>
                <div class="info-item"><strong>Thresholds:</strong> {th_str}</div>
                <div class="info-item"><strong>Sessions:</strong> {total_sessions} ({type_str})</div>
            </div>
        </header>

        {about_html}

        {upstream_banner_html}

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
                Metrics are based on segregated minutes and estimated energy, not session counts.
            </p>
            {overview_html}
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

        <section>
            <h2>Confidence Distribution</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Confidence scores for classified sessions (how well each session matches its device type)
                and exclusion confidence for unknown sessions (how confidently they were ruled out as known devices).
            </p>
            {confidence_html}
        </section>

        {activations_section_html}

        {glossary_html}

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

