"""
Cross-house insights computation.

Scans all houses' identification results and computes population-level
insights: device prevalence, temporal patterns, classification quality,
segregation effectiveness, anomalies, and device signatures.

All functions return structured dicts — no HTML generation here.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = logging.getLogger(__name__)

# Seasonal split (matching classification_quality.py convention)
WARM_MONTHS = {5, 6, 7, 8, 9, 10}
COOL_MONTHS = {1, 2, 3, 4, 11, 12}

DEVICE_TYPES = ['boiler', 'three_phase_device', 'central_ac', 'regular_ac',
                'recurring_pattern', 'unknown']
CLASSIFIED_TYPES = [dt for dt in DEVICE_TYPES if dt != 'unknown']

# Outlier thresholds (robust z-scores via median/MAD)
Z_EXTREME = 3.0
Z_WARNING = 2.0

# Top/bottom N for tables
TOP_N = 15


# ============================================================================
# Data loading
# ============================================================================

def load_all_house_data(
    experiment_dir: Path,
    house_ids: List[str],
    show_progress: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load device_sessions JSON for every house.

    Returns:
        {house_id: parsed JSON dict} for each house that has a valid JSON.
    """
    experiment_dir = Path(experiment_dir)
    sessions_dir = experiment_dir / 'device_sessions'

    result = {}
    it = house_ids
    if show_progress and _HAS_TQDM:
        it = _tqdm(house_ids, desc='Loading house data', unit='house')

    for hid in it:
        path = sessions_dir / f'device_sessions_{hid}.json'
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            result[hid] = data
        except Exception as e:
            logger.warning(f'Failed to load {path}: {e}')

    logger.info(f'Loaded {len(result)}/{len(house_ids)} house JSONs')
    return result


def load_segregation_data(
    experiment_dir: Path,
    house_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Load evaluation summary CSVs for segregation metrics.

    Returns:
        {house_id: {final_segregated_pct: {w1, w2, w3}, avg_segregated_pct: float}}
    """
    import csv

    experiment_dir = Path(experiment_dir)
    summaries_dir = experiment_dir / 'evaluation_summaries'
    if not summaries_dir.exists():
        logger.info('No evaluation_summaries directory found — skipping segregation data')
        return {}

    result = {}
    for hid in house_ids:
        path = summaries_dir / f'dynamic_evaluation_summary_{hid}.csv'
        if not path.exists():
            continue
        try:
            rows = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)

            if not rows:
                continue

            # Find the last iteration per phase
            max_run = max(int(r['run_number']) for r in rows)
            final_pct = {}
            for r in rows:
                if int(r['run_number']) == max_run:
                    phase = r['phase']
                    final_pct[phase] = float(r['cumulative_segregated_pct'])

            if final_pct:
                avg = np.mean(list(final_pct.values()))
                result[hid] = {
                    'final_segregated_pct': final_pct,
                    'avg_segregated_pct': round(float(avg), 2),
                }
        except Exception as e:
            logger.warning(f'Failed to load segregation data for {hid}: {e}')

    logger.info(f'Loaded segregation data for {len(result)}/{len(house_ids)} houses')
    return result


# ============================================================================
# Main entry point
# ============================================================================

def compute_all_insights(
    house_data: Dict[str, Dict],
    segregation_data: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Compute all insight categories from loaded data.

    Returns dict with keys: device_discovery, temporal_patterns,
    classification_quality, segregation_effectiveness, anomalies,
    device_signatures, top_findings.
    """
    n_houses = len(house_data)
    logger.info(f'Computing insights for {n_houses} houses')

    insights = {
        'n_houses': n_houses,
        'house_ids': sorted(house_data.keys()),
    }

    insights['device_discovery'] = _compute_device_discovery(house_data)
    insights['temporal_patterns'] = _compute_temporal_patterns(house_data)
    insights['classification_quality'] = _compute_classification_insights(house_data)

    if segregation_data:
        insights['segregation_effectiveness'] = _compute_segregation_insights(
            segregation_data)
    else:
        insights['segregation_effectiveness'] = None

    insights['anomalies'] = _compute_anomalies(house_data, segregation_data)
    insights['device_signatures'] = _compute_device_signatures(house_data)

    # G. House behavioral clustering
    insights['house_clusters'] = _compute_house_clusters(house_data, segregation_data)

    # Auto-generate the top findings dashboard
    insights['top_findings'] = _generate_top_findings(insights)

    return insights


# ============================================================================
# A. Device Discovery Overview
# ============================================================================

def _compute_device_discovery(house_data: Dict[str, Dict]) -> Dict[str, Any]:
    n_houses = len(house_data)

    # Prevalence: how many houses have at least one session of each type
    prevalence = {}
    for dt in DEVICE_TYPES:
        houses_with = []
        for hid, data in house_data.items():
            count = data.get('summary', {}).get('by_device_type', {}).get(dt, {}).get('count', 0)
            if count > 0:
                houses_with.append(hid)
        prevalence[dt] = {
            'count': len(houses_with),
            'pct': round(len(houses_with) / n_houses * 100, 1) if n_houses else 0,
            'house_ids': houses_with,
        }

    # Houses missing boiler
    missing_boiler = [hid for hid in house_data
                      if hid not in prevalence.get('boiler', {}).get('house_ids', [])]

    # Houses missing any AC (neither central nor regular)
    ac_houses = set(prevalence.get('central_ac', {}).get('house_ids', []))
    ac_houses.update(prevalence.get('regular_ac', {}).get('house_ids', []))
    missing_ac = [hid for hid in house_data if hid not in ac_houses]

    # Session count distribution
    session_counts = {}
    for hid, data in house_data.items():
        session_counts[hid] = data.get('summary', {}).get('total_sessions', 0)

    count_values = list(session_counts.values())
    count_dist = _robust_distribution(count_values) if count_values else {}

    # Outlier detection
    outlier_high = []
    outlier_low = []
    if count_dist.get('mad', 0) > 0:
        median = count_dist['median']
        mad = count_dist['mad']
        for hid, cnt in session_counts.items():
            z = abs(cnt - median) / mad
            if cnt > median and z > Z_WARNING:
                outlier_high.append({'house_id': hid, 'count': cnt, 'z_score': round(z, 1)})
            elif cnt < median and z > Z_WARNING:
                outlier_low.append({'house_id': hid, 'count': cnt, 'z_score': round(z, 1)})

    outlier_high.sort(key=lambda x: x['z_score'], reverse=True)
    outlier_low.sort(key=lambda x: x['z_score'], reverse=True)

    # Per-house device count breakdown
    per_house_counts = {}
    for hid, data in house_data.items():
        by_type = data.get('summary', {}).get('by_device_type', {})
        per_house_counts[hid] = {
            dt: by_type.get(dt, {}).get('count', 0) for dt in DEVICE_TYPES
        }

    return {
        'prevalence': prevalence,
        'missing_boiler': sorted(missing_boiler),
        'missing_ac': sorted(missing_ac),
        'session_count_distribution': count_dist,
        'session_counts': session_counts,
        'outlier_high_count': outlier_high[:TOP_N],
        'outlier_low_count': outlier_low[:TOP_N],
        'per_house_counts': per_house_counts,
    }


# ============================================================================
# B. Temporal Patterns
# ============================================================================

def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return None


def _compute_temporal_patterns(house_data: Dict[str, Dict]) -> Dict[str, Any]:
    # Hour-of-day counts per device type
    hourly = {dt: [0] * 24 for dt in DEVICE_TYPES}
    # Day-of-week counts (0=Mon..6=Sun)
    weekday = {dt: [0] * 7 for dt in DEVICE_TYPES}
    # Monthly counts (index 0-11 for Jan-Dec)
    monthly = {dt: [0] * 12 for dt in DEVICE_TYPES}

    # Per-house boiler activation hours (for regularity)
    boiler_hours_per_house = defaultdict(list)

    for hid, data in house_data.items():
        for session in data.get('sessions', []):
            dt_type = session.get('device_type', 'unknown')
            if dt_type not in hourly:
                dt_type = 'unknown'

            ts = _parse_iso(session.get('start'))
            if ts is None:
                continue

            hourly[dt_type][ts.hour] += 1
            weekday[dt_type][ts.weekday()] += 1
            monthly[dt_type][ts.month - 1] += 1

            if dt_type == 'boiler':
                boiler_hours_per_house[hid].append(ts.hour + ts.minute / 60.0)

    # Seasonal totals
    seasonal = {}
    for dt in DEVICE_TYPES:
        warm = sum(monthly[dt][m - 1] for m in WARM_MONTHS)
        cool = sum(monthly[dt][m - 1] for m in COOL_MONTHS)
        total = warm + cool
        seasonal[dt] = {
            'warm': warm,
            'cool': cool,
            'ratio': round(warm / cool, 2) if cool > 0 else None,
        }

    # Boiler regularity: std of activation hours
    regularity = {}
    for hid, hours in boiler_hours_per_house.items():
        if len(hours) >= 5:
            # Handle circular nature of hours (wrap-around at 24)
            arr = np.array(hours)
            std = float(np.std(arr))
            regularity[hid] = {'std_hours': round(std, 2), 'n_activations': len(hours)}

    # Sort by regularity
    most_regular = sorted(
        [{'house_id': h, **v} for h, v in regularity.items()],
        key=lambda x: x['std_hours']
    )[:TOP_N]

    most_irregular = sorted(
        [{'house_id': h, **v} for h, v in regularity.items()],
        key=lambda x: x['std_hours'],
        reverse=True
    )[:TOP_N]

    # Peak hours per device type
    peak_hours = {}
    for dt in DEVICE_TYPES:
        if sum(hourly[dt]) > 0:
            peak_h = int(np.argmax(hourly[dt]))
            peak_hours[dt] = {'hour': peak_h, 'count': hourly[dt][peak_h]}

    return {
        'hourly': hourly,
        'weekday': weekday,
        'monthly': monthly,
        'seasonal': seasonal,
        'peak_hours': peak_hours,
        'boiler_regularity': regularity,
        'most_regular_houses': most_regular,
        'most_irregular_houses': most_irregular,
    }


# ============================================================================
# C. Classification Quality
# ============================================================================

def _compute_classification_insights(house_data: Dict[str, Dict]) -> Dict[str, Any]:
    all_confidences = []
    per_device_confidences = defaultdict(list)
    per_house_avg_conf = {}
    per_house_unknown_pct = {}
    borderline_by_device = defaultdict(int)
    borderline_total = 0

    for hid, data in house_data.items():
        sessions = data.get('sessions', [])
        if not sessions:
            per_house_avg_conf[hid] = None
            per_house_unknown_pct[hid] = 100.0
            continue

        house_confs = []
        total_min = 0
        unknown_min = 0

        for s in sessions:
            dt_type = s.get('device_type', 'unknown')
            conf = s.get('confidence', 0)
            dur = s.get('duration_minutes', 0)

            total_min += dur
            if dt_type == 'unknown':
                unknown_min += dur
            else:
                house_confs.append(conf)
                per_device_confidences[dt_type].append(conf)

            all_confidences.append(conf)

            # Borderline: confidence 0.3-0.5
            if 0.3 <= conf <= 0.5:
                borderline_total += 1
                borderline_by_device[dt_type] += 1

        per_house_avg_conf[hid] = round(float(np.mean(house_confs)), 3) if house_confs else None
        per_house_unknown_pct[hid] = round(unknown_min / total_min * 100, 1) if total_min > 0 else 100.0

    # Confidence histogram (10 bins from 0 to 1)
    if all_confidences:
        hist_counts, hist_edges = np.histogram(all_confidences, bins=10, range=(0, 1))
        conf_histogram = {
            'counts': hist_counts.tolist(),
            'edges': hist_edges.tolist(),
        }
    else:
        conf_histogram = {'counts': [], 'edges': []}

    # Per-device average confidence
    per_device_avg = {}
    for dt, confs in per_device_confidences.items():
        per_device_avg[dt] = round(float(np.mean(confs)), 3) if confs else None

    # Hardest to classify (lowest average confidence, excluding unknown)
    classified_avgs = {k: v for k, v in per_device_avg.items()
                       if v is not None and k != 'unknown'}
    hardest = min(classified_avgs, key=classified_avgs.get) if classified_avgs else None

    # Low confidence houses (bottom 10%)
    valid_confs = {h: c for h, c in per_house_avg_conf.items() if c is not None}
    if valid_confs:
        threshold = np.percentile(list(valid_confs.values()), 10)
        low_conf_houses = sorted(
            [{'house_id': h, 'avg_confidence': c}
             for h, c in valid_confs.items() if c <= threshold],
            key=lambda x: x['avg_confidence']
        )[:TOP_N]
    else:
        low_conf_houses = []

    # High unknown houses
    high_unknown = sorted(
        [{'house_id': h, 'unknown_pct': p}
         for h, p in per_house_unknown_pct.items() if p > 50],
        key=lambda x: x['unknown_pct'],
        reverse=True
    )[:TOP_N]

    return {
        'confidence_histogram': conf_histogram,
        'per_house_avg_confidence': per_house_avg_conf,
        'per_house_unknown_pct': per_house_unknown_pct,
        'per_device_avg_confidence': per_device_avg,
        'hardest_to_classify': hardest,
        'low_confidence_houses': low_conf_houses,
        'high_unknown_houses': high_unknown,
        'borderline_total': borderline_total,
        'borderline_by_device': dict(borderline_by_device),
        'total_sessions': len(all_confidences),
    }


# ============================================================================
# D. Segregation Effectiveness
# ============================================================================

def _compute_segregation_insights(
    segregation_data: Dict[str, Dict],
) -> Dict[str, Any]:
    if not segregation_data:
        return {}

    avg_pcts = []
    phase_pcts = {'w1': [], 'w2': [], 'w3': []}
    per_house = {}

    for hid, sd in segregation_data.items():
        avg = sd.get('avg_segregated_pct', 0)
        avg_pcts.append(avg)
        per_house[hid] = avg

        for phase, pct in sd.get('final_segregated_pct', {}).items():
            if phase in phase_pcts:
                phase_pcts[phase].append(pct)

    dist = _robust_distribution(avg_pcts) if avg_pcts else {}

    per_phase_median = {}
    for phase, vals in phase_pcts.items():
        per_phase_median[phase] = round(float(np.median(vals)), 2) if vals else 0

    sorted_houses = sorted(per_house.items(), key=lambda x: x[1])
    bottom = [{'house_id': h, 'avg_pct': round(p, 2)} for h, p in sorted_houses[:TOP_N]]
    top = [{'house_id': h, 'avg_pct': round(p, 2)} for h, p in sorted_houses[-TOP_N:]]
    top.reverse()

    # Phase imbalance: houses where max-min phase difference > 10%
    imbalance = []
    for hid, sd in segregation_data.items():
        pcts = list(sd.get('final_segregated_pct', {}).values())
        if len(pcts) >= 2:
            diff = max(pcts) - min(pcts)
            if diff > 10:
                imbalance.append({'house_id': hid, 'max_diff': round(diff, 1),
                                  'phases': sd.get('final_segregated_pct', {})})
    imbalance.sort(key=lambda x: x['max_diff'], reverse=True)

    return {
        'distribution': dist,
        'per_phase_median': per_phase_median,
        'bottom_houses': bottom,
        'top_houses': top,
        'phase_imbalance_houses': imbalance[:TOP_N],
        'per_house': per_house,
    }


# ============================================================================
# E. Anomalies & Outliers
# ============================================================================

def _compute_anomalies(
    house_data: Dict[str, Dict],
    segregation_data: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    unusual_combinations = []
    high_spike_houses = []
    zero_session_houses = []

    spike_counts = {}

    for hid, data in house_data.items():
        summary = data.get('summary', {})
        by_type = summary.get('by_device_type', {})
        total = summary.get('total_sessions', 0)

        # Zero-session houses
        if total == 0:
            zero_session_houses.append(hid)
            continue

        has_boiler = by_type.get('boiler', {}).get('count', 0) > 0
        has_central = by_type.get('central_ac', {}).get('count', 0) > 0
        has_regular = by_type.get('regular_ac', {}).get('count', 0) > 0
        has_recurring = by_type.get('recurring_pattern', {}).get('count', 0) > 0
        unknown_count = by_type.get('unknown', {}).get('count', 0)

        # Unusual: central AC but no regular AC
        if has_central and not has_regular:
            unusual_combinations.append({
                'house_id': hid,
                'description': 'Central AC detected but no regular AC',
            })

        # Unusual: very high unknown ratio (>70% of sessions)
        if total > 10 and unknown_count / total > 0.7:
            unusual_combinations.append({
                'house_id': hid,
                'description': f'Very high unknown rate: {unknown_count}/{total} sessions '
                               f'({unknown_count / total * 100:.0f}%)',
            })

        # Spike counts
        spike_filter = data.get('spike_filter', {})
        sc = spike_filter.get('spike_count', 0)
        spike_counts[hid] = sc

    # High spike houses (top 5%)
    if spike_counts:
        threshold_95 = np.percentile(list(spike_counts.values()), 95)
        high_spike_houses = sorted(
            [{'house_id': h, 'spike_count': c}
             for h, c in spike_counts.items() if c >= threshold_95],
            key=lambda x: x['spike_count'],
            reverse=True
        )[:TOP_N]

    # Composite outlier score (multi-dimensional)
    composite_scores = {}
    for hid, data in house_data.items():
        score = 0
        reasons = []

        # Dimension 1: session count (very low or very high)
        total = data.get('summary', {}).get('total_sessions', 0)
        all_totals = [d.get('summary', {}).get('total_sessions', 0) for d in house_data.values()]
        if all_totals:
            z = _z_score_robust(total, all_totals)
            if abs(z) > Z_WARNING:
                score += abs(z)
                reasons.append(f'Session count z={z:.1f} ({total} sessions)')

        # Dimension 2: unknown rate
        sessions = data.get('sessions', [])
        if sessions:
            total_min = sum(s.get('duration_minutes', 0) for s in sessions)
            unknown_min = sum(s.get('duration_minutes', 0) for s in sessions
                             if s.get('device_type') == 'unknown')
            unknown_pct = unknown_min / total_min * 100 if total_min > 0 else 0
            all_unk = []
            for d in house_data.values():
                ss = d.get('sessions', [])
                tm = sum(s.get('duration_minutes', 0) for s in ss)
                um = sum(s.get('duration_minutes', 0) for s in ss
                         if s.get('device_type') == 'unknown')
                all_unk.append(um / tm * 100 if tm > 0 else 0)
            z = _z_score_robust(unknown_pct, all_unk)
            if abs(z) > Z_WARNING:
                score += abs(z)
                reasons.append(f'Unknown rate z={z:.1f} ({unknown_pct:.0f}%)')

        # Dimension 3: spike count
        sc = spike_counts.get(hid, 0)
        if spike_counts:
            z = _z_score_robust(sc, list(spike_counts.values()))
            if abs(z) > Z_WARNING:
                score += abs(z)
                reasons.append(f'Spike count z={z:.1f} ({sc} spikes)')

        # Dimension 4: segregation (if available)
        if segregation_data and hid in segregation_data:
            seg_pct = segregation_data[hid].get('avg_segregated_pct', 0)
            all_seg = [sd.get('avg_segregated_pct', 0) for sd in segregation_data.values()]
            z = _z_score_robust(seg_pct, all_seg)
            if abs(z) > Z_WARNING:
                score += abs(z)
                reasons.append(f'Segregation z={z:.1f} ({seg_pct:.1f}%)')

        if score > 0:
            composite_scores[hid] = {'score': round(score, 2), 'reasons': reasons}

    top_outliers = sorted(
        [{'house_id': h, **v} for h, v in composite_scores.items()],
        key=lambda x: x['score'],
        reverse=True
    )[:TOP_N]

    return {
        'unusual_combinations': unusual_combinations,
        'high_spike_houses': high_spike_houses,
        'zero_session_houses': sorted(zero_session_houses),
        'top_outliers': top_outliers,
        'spike_counts': spike_counts,
    }


# ============================================================================
# F. Device Signatures
# ============================================================================

def _compute_device_signatures(house_data: Dict[str, Dict]) -> Dict[str, Any]:
    magnitude_by_device = defaultdict(list)
    duration_by_device = defaultdict(list)
    phase_preference = defaultdict(lambda: defaultdict(int))

    # Recurring pattern info
    recurring_patterns = defaultdict(lambda: {
        'house_ids': set(), 'total_sessions': 0,
        'magnitudes': [], 'durations': [],
    })

    for hid, data in house_data.items():
        for session in data.get('sessions', []):
            dt_type = session.get('device_type', 'unknown')

            mag = session.get('avg_cycle_magnitude_w')
            dur = session.get('duration_minutes')
            phases = session.get('phases', [])

            if mag is not None and mag > 0:
                magnitude_by_device[dt_type].append(mag)
            if dur is not None and dur > 0:
                duration_by_device[dt_type].append(dur)

            # Phase preference
            if len(phases) == 1:
                phase_preference[dt_type][phases[0]] += 1
            elif len(phases) > 1:
                phase_preference[dt_type]['multi'] += 1

            # Recurring patterns — group by global_pattern_name
            if dt_type == 'recurring_pattern':
                cb = session.get('confidence_breakdown', {})
                gname = cb.get('global_pattern_name')
                if gname:
                    rp = recurring_patterns[gname]
                    rp['house_ids'].add(hid)
                    rp['total_sessions'] += 1
                    if mag:
                        rp['magnitudes'].append(mag)
                    if dur:
                        rp['durations'].append(dur)

    # Summarize magnitude/duration distributions
    mag_summary = {}
    dur_summary = {}
    for dt in DEVICE_TYPES:
        vals = magnitude_by_device.get(dt, [])
        if vals:
            mag_summary[dt] = {
                'median': round(float(np.median(vals)), 0),
                'q25': round(float(np.percentile(vals, 25)), 0),
                'q75': round(float(np.percentile(vals, 75)), 0),
                'min': round(float(np.min(vals)), 0),
                'max': round(float(np.max(vals)), 0),
                'n': len(vals),
            }
        vals = duration_by_device.get(dt, [])
        if vals:
            dur_summary[dt] = {
                'median': round(float(np.median(vals)), 1),
                'q25': round(float(np.percentile(vals, 25)), 1),
                'q75': round(float(np.percentile(vals, 75)), 1),
                'min': round(float(np.min(vals)), 1),
                'max': round(float(np.max(vals)), 1),
                'n': len(vals),
            }

    # Phase preference summary
    phase_pref_summary = {}
    for dt in DEVICE_TYPES:
        if dt in phase_preference:
            phase_pref_summary[dt] = dict(phase_preference[dt])

    # Recurring patterns summary
    rp_list = []
    for gname, info in recurring_patterns.items():
        rp_list.append({
            'name': gname,
            'house_count': len(info['house_ids']),
            'total_sessions': info['total_sessions'],
            'avg_magnitude': round(float(np.mean(info['magnitudes'])), 0) if info['magnitudes'] else 0,
            'avg_duration': round(float(np.mean(info['durations'])), 1) if info['durations'] else 0,
        })
    rp_list.sort(key=lambda x: x['house_count'], reverse=True)

    return {
        'magnitude_by_device': mag_summary,
        'duration_by_device': dur_summary,
        'magnitude_raw': {dt: vals for dt, vals in magnitude_by_device.items()},
        'duration_raw': {dt: vals for dt, vals in duration_by_device.items()},
        'phase_preference': phase_pref_summary,
        'recurring_patterns': rp_list,
    }


# ============================================================================
# G. Research Insights — Behavioral Similarity & Outcome Divergence
# ============================================================================

def _compute_house_clusters(
    house_data: Dict[str, Dict],
    segregation_data: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Research-oriented analysis: find houses with similar behavior but different
    pipeline outcomes, identify success/failure predictors, and characterize
    failure patterns.

    Three sub-analyses:
    1. Similar-but-divergent: house pairs with similar input but different outcomes
    2. Success predictors: which features correlate with pipeline success
    3. Failure patterns: what struggling houses have in common
    """
    n_houses = len(house_data)
    if n_houses < 4:
        return {'error': 'Too few houses for analysis'}

    # ── Build per-house profiles ──────────────────────────────────
    profiles = _build_house_profiles(house_data, segregation_data)

    # ── 1. Similar-but-divergent pairs ────────────────────────────
    divergent_pairs = _find_divergent_pairs(profiles)

    # ── 2. Success predictors ─────────────────────────────────────
    success_analysis = _analyze_success_predictors(profiles)

    # ── 3. Failure patterns ───────────────────────────────────────
    failure_analysis = _analyze_failure_patterns(profiles, house_data)

    # ── 4. House comparison matrix ────────────────────────────────
    comparison_matrix = _build_comparison_matrix(profiles)

    return {
        'profiles': profiles,
        'divergent_pairs': divergent_pairs,
        'success_analysis': success_analysis,
        'failure_analysis': failure_analysis,
        'comparison_matrix': comparison_matrix,
    }


def _build_house_profiles(
    house_data: Dict[str, Dict],
    segregation_data: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """Build behavioral profile for each house.

    Separates features into INPUT characteristics (what the house looks like)
    and OUTCOME metrics (how well the pipeline performed).
    """
    profiles = {}

    for hid, data in house_data.items():
        summary = data.get('summary', {})
        by_type = summary.get('by_device_type', {})
        sessions = data.get('sessions', [])
        spike_filter = data.get('spike_filter', {})

        boiler_count = by_type.get('boiler', {}).get('count', 0)
        central_count = by_type.get('central_ac', {}).get('count', 0)
        regular_count = by_type.get('regular_ac', {}).get('count', 0)
        recurring_count = by_type.get('recurring_pattern', {}).get('count', 0)
        unknown_count = by_type.get('unknown', {}).get('count', 0)
        total = summary.get('total_sessions', 0)

        # Duration-weighted unknown percentage
        total_min = sum(s.get('duration_minutes', 0) for s in sessions)
        unknown_min = sum(s.get('duration_minutes', 0) for s in sessions
                         if s.get('device_type') == 'unknown')
        unknown_pct = unknown_min / total_min * 100 if total_min > 0 else 100

        # Average confidence of classified sessions
        classified_confs = [s.get('confidence', 0) for s in sessions
                           if s.get('device_type') != 'unknown']
        avg_conf = float(np.mean(classified_confs)) if classified_confs else 0

        # Spike ratio
        spike_count = spike_filter.get('spike_count', 0)
        kept_count = spike_filter.get('kept_count', 0)
        total_detected = spike_count + kept_count
        spike_ratio = spike_count / total_detected if total_detected > 0 else 0

        # Average magnitude across all sessions
        mags = [s.get('avg_cycle_magnitude_w', 0) for s in sessions
                if s.get('avg_cycle_magnitude_w')]
        avg_magnitude = float(np.mean(mags)) if mags else 0

        # Duration profile
        durs = [s.get('duration_minutes', 0) for s in sessions if s.get('duration_minutes')]
        median_duration = float(np.median(durs)) if durs else 0

        # Segregation
        seg_pct = 0
        if segregation_data and hid in segregation_data:
            seg_pct = segregation_data[hid].get('avg_segregated_pct', 0)

        # Classification rate (non-unknown / total)
        classified_pct = (total - unknown_count) / total * 100 if total > 0 else 0

        # "Pipeline success score" — composite metric
        # Higher = pipeline did better on this house
        success_score = (
            0.35 * min(classified_pct / 100, 1) +         # classification rate
            0.25 * avg_conf +                               # confidence quality
            0.25 * min(seg_pct / 20, 1) +                  # segregation (20% = perfect)
            0.15 * (1 - min(unknown_pct / 100, 1))         # inverse unknown rate
        )

        profiles[hid] = {
            # INPUT characteristics (house behavior)
            'total_sessions': total,
            'total_events_detected': total_detected,
            'avg_magnitude': round(avg_magnitude, 0),
            'median_duration': round(median_duration, 1),
            'spike_ratio': round(spike_ratio, 3),
            'has_boiler': boiler_count > 0,
            'has_ac': (central_count + regular_count) > 0,
            'has_recurring': recurring_count > 0,
            'device_variety': sum([boiler_count > 0, central_count > 0,
                                   regular_count > 0, recurring_count > 0]),
            # OUTCOME metrics (pipeline performance)
            'classified_pct': round(classified_pct, 1),
            'unknown_pct': round(unknown_pct, 1),
            'avg_confidence': round(avg_conf, 3),
            'segregation_pct': round(seg_pct, 2),
            'success_score': round(success_score, 3),
            # Raw counts for display
            'boiler_count': boiler_count,
            'ac_count': central_count + regular_count,
            'recurring_count': recurring_count,
            'unknown_count': unknown_count,
        }

    return profiles


def _find_divergent_pairs(profiles: Dict[str, Dict]) -> List[Dict]:
    """Find house pairs that are behaviorally similar but have divergent
    pipeline outcomes.

    Similarity is based on INPUT features (magnitude, sessions, spike ratio).
    Divergence is based on OUTCOME metrics (success_score, classified_pct).
    """
    from scipy.spatial.distance import pdist, squareform

    house_ids = sorted(profiles.keys())
    n = len(house_ids)
    if n < 2:
        return []

    # Build input feature matrix (only behavioral characteristics)
    input_features = ['total_sessions', 'avg_magnitude', 'median_duration',
                      'spike_ratio', 'total_events_detected']
    input_matrix = np.zeros((n, len(input_features)))
    for i, hid in enumerate(house_ids):
        for j, feat in enumerate(input_features):
            input_matrix[i, j] = profiles[hid].get(feat, 0)

    # Normalize
    means = input_matrix.mean(axis=0)
    stds = input_matrix.std(axis=0)
    stds[stds < 1e-6] = 1
    norm_input = (input_matrix - means) / stds

    # Pairwise behavioral similarity
    input_dists = squareform(pdist(norm_input, metric='euclidean'))

    # Find pairs: similar input (low distance) but divergent outcome
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            h1, h2 = house_ids[i], house_ids[j]
            behavioral_dist = input_dists[i, j]
            outcome_diff = abs(profiles[h1]['success_score'] -
                              profiles[h2]['success_score'])

            # Similar behavior: distance in bottom 30th percentile
            # Divergent outcome: success_score diff > 0.15
            if outcome_diff > 0.1:
                pairs.append({
                    'house_a': h1,
                    'house_b': h2,
                    'behavioral_similarity': round(1 / (1 + behavioral_dist), 3),
                    'behavioral_distance': round(behavioral_dist, 3),
                    'outcome_divergence': round(outcome_diff, 3),
                    'score_a': profiles[h1]['success_score'],
                    'score_b': profiles[h2]['success_score'],
                    'better': h1 if profiles[h1]['success_score'] > profiles[h2]['success_score'] else h2,
                    'worse': h2 if profiles[h1]['success_score'] > profiles[h2]['success_score'] else h1,
                    # What differs in the outcome?
                    'diff_classified': round(profiles[h1]['classified_pct'] - profiles[h2]['classified_pct'], 1),
                    'diff_segregation': round(profiles[h1]['segregation_pct'] - profiles[h2]['segregation_pct'], 1),
                    'diff_confidence': round(profiles[h1]['avg_confidence'] - profiles[h2]['avg_confidence'], 3),
                })

    # Sort by: high similarity × high divergence (most interesting first)
    pairs.sort(key=lambda p: p['behavioral_similarity'] * p['outcome_divergence'],
               reverse=True)

    return pairs[:20]


def _analyze_success_predictors(profiles: Dict[str, Dict]) -> Dict[str, Any]:
    """Identify which house characteristics correlate with pipeline success.

    Uses simple correlation analysis between input features and the
    composite success score.
    """
    house_ids = sorted(profiles.keys())
    if len(house_ids) < 5:
        return {}

    success_scores = np.array([profiles[h]['success_score'] for h in house_ids])

    input_features = {
        'total_sessions': 'Total Sessions',
        'avg_magnitude': 'Avg Magnitude (W)',
        'median_duration': 'Median Duration (min)',
        'spike_ratio': 'Spike Ratio',
        'total_events_detected': 'Total Events Detected',
        'device_variety': 'Device Type Variety',
    }

    correlations = []
    for feat, label in input_features.items():
        values = np.array([profiles[h].get(feat, 0) for h in house_ids], dtype=float)
        if np.std(values) < 1e-6:
            continue
        # Pearson correlation
        corr = float(np.corrcoef(values, success_scores)[0, 1])
        correlations.append({
            'feature': feat,
            'label': label,
            'correlation': round(corr, 3),
            'abs_correlation': round(abs(corr), 3),
            'direction': 'positive' if corr > 0 else 'negative',
            'interpretation': _interpret_correlation(feat, corr),
        })

    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Divide houses into success tiers
    sorted_houses = sorted(house_ids, key=lambda h: profiles[h]['success_score'])
    n = len(sorted_houses)
    tier_size = max(1, n // 3)

    low_houses = sorted_houses[:tier_size]
    mid_houses = sorted_houses[tier_size:n - tier_size]
    high_houses = sorted_houses[n - tier_size:]

    # Compare tiers
    tier_comparison = {}
    for feat, label in input_features.items():
        low_vals = [profiles[h].get(feat, 0) for h in low_houses]
        high_vals = [profiles[h].get(feat, 0) for h in high_houses]
        tier_comparison[feat] = {
            'label': label,
            'low_tier_mean': round(float(np.mean(low_vals)), 1) if low_vals else 0,
            'high_tier_mean': round(float(np.mean(high_vals)), 1) if high_vals else 0,
            'difference_pct': _safe_pct_diff(
                float(np.mean(high_vals)) if high_vals else 0,
                float(np.mean(low_vals)) if low_vals else 0),
        }

    return {
        'correlations': correlations,
        'tier_comparison': tier_comparison,
        'low_tier': {
            'house_ids': sorted(low_houses),
            'avg_success': round(float(np.mean([profiles[h]['success_score']
                                                 for h in low_houses])), 3),
        },
        'high_tier': {
            'house_ids': sorted(high_houses),
            'avg_success': round(float(np.mean([profiles[h]['success_score']
                                                 for h in high_houses])), 3),
        },
    }


def _analyze_failure_patterns(
    profiles: Dict[str, Dict],
    house_data: Dict[str, Dict],
) -> Dict[str, Any]:
    """Characterize what struggling houses have in common.

    Identifies specific failure modes and groups houses by failure type.
    """
    failure_modes = {
        'high_unknown': {
            'label': 'High Unknown Rate (>60%)',
            'description': 'Pipeline detects events but cannot classify them',
            'houses': [],
        },
        'low_segregation': {
            'label': 'Low Segregation (<3%)',
            'description': 'Pipeline barely explains any power consumption',
            'houses': [],
        },
        'low_confidence': {
            'label': 'Low Confidence (<0.4)',
            'description': 'Classifications exist but are unreliable',
            'houses': [],
        },
        'few_sessions': {
            'label': 'Very Few Sessions (<10)',
            'description': 'Pipeline finds almost no device activations',
            'houses': [],
        },
        'high_spike_ratio': {
            'label': 'High Spike Ratio (>50%)',
            'description': 'Most detected events are transient noise',
            'houses': [],
        },
    }

    for hid, p in profiles.items():
        if p['unknown_pct'] > 60:
            failure_modes['high_unknown']['houses'].append(hid)
        if p['segregation_pct'] < 3:
            failure_modes['low_segregation']['houses'].append(hid)
        if p['avg_confidence'] < 0.4 and p['total_sessions'] > 5:
            failure_modes['low_confidence']['houses'].append(hid)
        if p['total_sessions'] < 10:
            failure_modes['few_sessions']['houses'].append(hid)
        if p['spike_ratio'] > 0.5:
            failure_modes['high_spike_ratio']['houses'].append(hid)

    # For each failure mode, compute the average profile of affected houses
    for mode_key, mode in failure_modes.items():
        houses = mode['houses']
        if not houses:
            mode['avg_profile'] = {}
            continue
        mode['avg_profile'] = {
            'n_houses': len(houses),
            'avg_sessions': round(float(np.mean(
                [profiles[h]['total_sessions'] for h in houses])), 0),
            'avg_magnitude': round(float(np.mean(
                [profiles[h]['avg_magnitude'] for h in houses])), 0),
            'avg_classified_pct': round(float(np.mean(
                [profiles[h]['classified_pct'] for h in houses])), 1),
            'avg_segregation': round(float(np.mean(
                [profiles[h]['segregation_pct'] for h in houses])), 1),
        }

    # Houses with multiple failure modes (compound failures)
    house_failures = defaultdict(list)
    for mode_key, mode in failure_modes.items():
        for hid in mode['houses']:
            house_failures[hid].append(mode['label'])

    compound_failures = [
        {'house_id': hid, 'failure_modes': modes, 'n_modes': len(modes)}
        for hid, modes in house_failures.items() if len(modes) >= 2
    ]
    compound_failures.sort(key=lambda x: x['n_modes'], reverse=True)

    # Success stories — houses with no failure modes at all
    all_failing = set()
    for mode in failure_modes.values():
        all_failing.update(mode['houses'])
    success_houses = [hid for hid in profiles if hid not in all_failing]

    return {
        'failure_modes': failure_modes,
        'compound_failures': compound_failures,
        'success_houses': sorted(success_houses),
        'n_struggling': len(all_failing),
        'n_healthy': len(success_houses),
    }


def _build_comparison_matrix(profiles: Dict[str, Dict]) -> Dict[str, Any]:
    """Build a comprehensive house comparison table for the report.

    Each house gets a single row with input features + outcome metrics +
    a success tier label, designed for sortable exploration.
    """
    house_ids = sorted(profiles.keys())
    if not house_ids:
        return {'rows': []}

    scores = [profiles[h]['success_score'] for h in house_ids]
    if scores:
        q33 = float(np.percentile(scores, 33))
        q66 = float(np.percentile(scores, 66))
    else:
        q33 = q66 = 0

    rows = []
    for hid in house_ids:
        p = profiles[hid]
        score = p['success_score']
        tier = 'high' if score >= q66 else ('mid' if score >= q33 else 'low')
        rows.append({
            'house_id': hid,
            'total_sessions': p['total_sessions'],
            'avg_magnitude': p['avg_magnitude'],
            'spike_ratio': p['spike_ratio'],
            'boiler': p['boiler_count'],
            'ac': p['ac_count'],
            'recurring': p['recurring_count'],
            'unknown': p['unknown_count'],
            'classified_pct': p['classified_pct'],
            'unknown_pct': p['unknown_pct'],
            'avg_confidence': p['avg_confidence'],
            'segregation_pct': p['segregation_pct'],
            'success_score': p['success_score'],
            'tier': tier,
        })

    rows.sort(key=lambda r: r['success_score'], reverse=True)

    return {'rows': rows, 'q33': round(q33, 3), 'q66': round(q66, 3)}


def _interpret_correlation(feature: str, corr: float) -> str:
    """Generate human-readable interpretation of a correlation."""
    strength = 'Strong' if abs(corr) > 0.5 else ('Moderate' if abs(corr) > 0.3 else 'Weak')
    direction = 'more' if corr > 0 else 'less'

    interpretations = {
        'total_sessions': f'{strength}: houses with {direction} sessions tend to have '
                          f'{"better" if corr > 0 else "worse"} pipeline results',
        'avg_magnitude': f'{strength}: {"higher" if corr > 0 else "lower"} power devices → '
                         f'{"better" if corr > 0 else "worse"} classification',
        'median_duration': f'{strength}: {"longer" if corr > 0 else "shorter"} sessions → '
                           f'{"easier" if corr > 0 else "harder"} to classify',
        'spike_ratio': f'{strength}: {"more" if corr > 0 else "fewer"} spikes → '
                       f'{"better" if corr > 0 else "worse"} results '
                       f'({"surprising" if corr > 0 else "expected"})',
        'total_events_detected': f'{strength}: {direction} raw events → '
                                  f'{"better" if corr > 0 else "worse"} outcomes',
        'device_variety': f'{strength}: {direction} device types → '
                          f'{"better" if corr > 0 else "worse"} outcomes',
    }
    return interpretations.get(feature, f'{strength} {"positive" if corr > 0 else "negative"} correlation')


def _safe_pct_diff(a: float, b: float) -> float:
    """Percentage difference between a and b, safe for zero."""
    if abs(b) < 1e-6:
        return 0
    return round((a - b) / abs(b) * 100, 1)


# ============================================================================
# Top Findings Generator
# ============================================================================

def _generate_top_findings(insights: Dict[str, Any]) -> List[Dict[str, str]]:
    """Auto-generate the most interesting findings for the dashboard."""
    findings = []
    n_houses = insights.get('n_houses', 0)
    if n_houses == 0:
        return findings

    # --- Device discovery findings ---
    dd = insights.get('device_discovery', {})
    prevalence = dd.get('prevalence', {})

    boiler_pct = prevalence.get('boiler', {}).get('pct', 0)
    findings.append({
        'title': f'{boiler_pct:.0f}% of houses have a boiler',
        'description': f'{prevalence.get("boiler", {}).get("count", 0)}/{n_houses} houses '
                       f'have at least one water heater session detected',
        'category': 'device_discovery',
        'severity': 'info',
    })

    ac_count = len(set(prevalence.get('central_ac', {}).get('house_ids', [])) |
                   set(prevalence.get('regular_ac', {}).get('house_ids', [])))
    findings.append({
        'title': f'{ac_count}/{n_houses} houses have AC detected',
        'description': f'{prevalence.get("central_ac", {}).get("count", 0)} central + '
                       f'{prevalence.get("regular_ac", {}).get("count", 0)} regular AC houses',
        'category': 'device_discovery',
        'severity': 'info',
    })

    # --- Temporal finding ---
    tp = insights.get('temporal_patterns', {})
    peak_boiler = tp.get('peak_hours', {}).get('boiler')
    if peak_boiler:
        findings.append({
            'title': f'Boiler peak hour: {peak_boiler["hour"]:02d}:00',
            'description': f'{peak_boiler["count"]} boiler activations at this hour '
                           f'across all houses',
            'category': 'temporal',
            'severity': 'info',
        })

    # --- AC seasonality ---
    seasonal_ac = tp.get('seasonal', {}).get('regular_ac', {})
    if seasonal_ac.get('ratio') is not None:
        findings.append({
            'title': f'AC warm/cool ratio: {seasonal_ac["ratio"]:.1f}x',
            'description': f'{seasonal_ac["warm"]} warm-season vs {seasonal_ac["cool"]} '
                           f'cool-season AC sessions',
            'category': 'temporal',
            'severity': 'info',
        })

    # --- Classification quality ---
    cq = insights.get('classification_quality', {})
    high_unknown = cq.get('high_unknown_houses', [])
    if high_unknown:
        findings.append({
            'title': f'{len(high_unknown)} houses have >50% unknown',
            'description': f'These houses have most session-minutes unclassified',
            'category': 'classification',
            'severity': 'warning',
        })

    hardest = cq.get('hardest_to_classify')
    if hardest:
        avg_c = cq.get('per_device_avg_confidence', {}).get(hardest, 0)
        findings.append({
            'title': f'Hardest to classify: {hardest}',
            'description': f'Average confidence {avg_c:.0%} — lowest among device types',
            'category': 'classification',
            'severity': 'warning',
        })

    # --- Segregation ---
    seg = insights.get('segregation_effectiveness')
    if seg and seg.get('distribution'):
        median_seg = seg['distribution'].get('median', 0)
        findings.append({
            'title': f'Median segregation: {median_seg:.1f}%',
            'description': f'Median of average segregated power across all houses',
            'category': 'segregation',
            'severity': 'info' if median_seg > 5 else 'warning',
        })

    # --- Anomalies ---
    anom = insights.get('anomalies', {})
    zero = anom.get('zero_session_houses', [])
    if zero:
        findings.append({
            'title': f'{len(zero)} houses with zero sessions',
            'description': f'Pipeline found no device sessions in these houses',
            'category': 'anomaly',
            'severity': 'warning' if len(zero) < 10 else 'critical',
        })

    # --- Research insights ---
    hc = insights.get('house_clusters', {})
    failure = hc.get('failure_analysis', {})
    n_struggling = failure.get('n_struggling', 0)
    n_healthy = failure.get('n_healthy', 0)
    if n_struggling > 0 or n_healthy > 0:
        findings.append({
            'title': f'{n_healthy} healthy, {n_struggling} struggling houses',
            'description': f'{n_healthy}/{n_houses} houses have no failure flags; '
                           f'{n_struggling} have at least one issue',
            'category': 'research',
            'severity': 'info' if n_struggling < n_houses * 0.3 else 'warning',
        })

    success = hc.get('success_analysis', {})
    corrs = success.get('correlations', [])
    if corrs:
        top_corr = corrs[0]
        findings.append({
            'title': f'Top predictor: {top_corr["label"]}',
            'description': f'{top_corr["interpretation"]} (r={top_corr["correlation"]:.2f})',
            'category': 'research',
            'severity': 'info',
        })

    divergent = hc.get('divergent_pairs', [])
    if divergent:
        top = divergent[0]
        findings.append({
            'title': f'Most divergent pair: {top["better"]} vs {top["worse"]}',
            'description': f'Similar behavior but success scores {top["score_a"]:.2f} vs '
                           f'{top["score_b"]:.2f}',
            'category': 'research',
            'severity': 'warning',
        })

    return findings


# ============================================================================
# Utilities
# ============================================================================

def _robust_distribution(values: List[float]) -> Dict[str, float]:
    """Compute robust distribution statistics (median, MAD)."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826  # scaled MAD
    return {
        'mean': round(float(np.mean(arr)), 2),
        'median': round(median, 2),
        'std': round(float(np.std(arr)), 2),
        'mad': round(mad, 2),
        'min': round(float(np.min(arr)), 2),
        'max': round(float(np.max(arr)), 2),
        'q25': round(float(np.percentile(arr, 25)), 2),
        'q75': round(float(np.percentile(arr, 75)), 2),
        'n': len(values),
    }


def _z_score_robust(value: float, all_values: List[float]) -> float:
    """Compute robust z-score using median/MAD."""
    if not all_values:
        return 0.0
    arr = np.array(all_values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826
    if mad < 1e-6:
        return 0.0
    return (value - median) / mad
