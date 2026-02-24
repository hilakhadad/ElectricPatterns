"""
Session-level JSON output builder.

Produces:
- ``device_sessions/device_sessions_{house_id}.json``  — new session-level format
- ``device_activations/device_activations_{house_id}.json`` — backward-compatible
  flat format so existing analysis reports keep working during transition.
"""
import json
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .session_classifier import ClassifiedSession
from .session_grouper import Session, MultiPhaseSession

logger = logging.getLogger(__name__)


def build_session_json(
    classified_sessions: Dict[str, List[ClassifiedSession]],
    house_id: str,
    threshold_schedule: List[int],
    experiment_dir: Path,
    device_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    session_gap_minutes: int = 30,
    spike_stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """Build and save session-level + backward-compatible JSON output.

    Args:
        classified_sessions: Output from :func:`classify_sessions`.
        house_id: House ID.
        threshold_schedule: [2000, 1500, 1100, 800].
        experiment_dir: Root experiment output directory.
        device_profiles: Optional per-minute power profiles from segmentation.
        session_gap_minutes: Gap used for session grouping.

    Returns:
        Path to the saved session JSON file.
    """
    # Build session entries
    sessions_list: List[dict] = []
    all_events_flat: List[dict] = []  # for backward-compatible JSON

    for device_type, classified_list in classified_sessions.items():
        for cs in classified_list:
            session_entry = _build_session_entry(cs, device_profiles)
            sessions_list.append(session_entry)

            # Flatten constituent events with device_type for backward compat
            for event in session_entry.get('constituent_events', []):
                flat = dict(event)
                flat['device_type'] = device_type
                flat['session_id'] = session_entry['session_id']
                flat['confidence'] = session_entry.get('confidence', 0)
                flat['match_type'] = 'matched'
                all_events_flat.append(flat)

    # Sort by start time
    sessions_list.sort(key=lambda s: s.get('start', '9999'))

    # Summary
    summary = _build_summary(classified_sessions)

    # --- Save session JSON ---
    sessions_dir = experiment_dir / "device_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / f"device_sessions_{house_id}.json"

    session_output = {
        'house_id': house_id,
        'generated_at': datetime.now().isoformat(),
        'threshold_schedule': threshold_schedule,
        'session_gap_minutes': session_gap_minutes,
        'summary': summary,
        'spike_filter': spike_stats or {},
        'sessions': sessions_list,
    }

    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(session_output, f, indent=2, default=_json_serializer)

    logger.info(f"Session JSON saved: {len(sessions_list)} sessions → {session_path}")

    # --- Save backward-compatible flat JSON ---
    _save_backward_compatible_json(
        all_events_flat, house_id, threshold_schedule, experiment_dir
    )

    return session_path


# ============================================================================
# Internal helpers
# ============================================================================

def _build_session_entry(cs: ClassifiedSession, device_profiles=None) -> dict:
    """Build a single session JSON entry."""
    session = cs.session

    if isinstance(session, MultiPhaseSession):
        phases = session.phases
        start = session.start
        end = session.end
        cycle_count = sum(ps.cycle_count for ps in session.phase_sessions.values())
        magnitudes = {p: ps.avg_magnitude for p, ps in session.phase_sessions.items()}
        avg_mag = session.total_magnitude / len(session.phases) if session.phases else 0
        events = []
        for ps in session.phase_sessions.values():
            events.extend(ps.events)
    else:
        phases = [session.phase]
        start = session.start
        end = session.end
        cycle_count = session.cycle_count
        magnitudes = {session.phase: session.avg_magnitude}
        avg_mag = session.avg_magnitude
        events = session.events

    total_dur = (end - start).total_seconds() / 60 if start and end else 0

    constituent = []
    for ev in sorted(events, key=lambda e: str(e.get('on_start', ''))):
        constituent.append(_serialize_event(ev, device_profiles))

    entry = {
        'session_id': session.session_id,
        'device_type': cs.device_type,
        'confidence': cs.confidence,
        'confidence_breakdown': cs.confidence_breakdown,
        'classification_reason': cs.reason,
        'start': _ts(start),
        'end': _ts(end),
        'duration_minutes': round(total_dur, 1),
        'phases': phases,
        'phase_presence': {p: 'V' if p in phases else 'X' for p in ['w1', 'w2', 'w3']},
        'cycle_count': cycle_count,
        'avg_cycle_magnitude_w': round(avg_mag, 1),
        'phase_magnitudes': {k: round(v, 1) for k, v in magnitudes.items()},
        'constituent_events': constituent,
    }
    return entry


def _serialize_event(ev: dict, device_profiles=None) -> dict:
    """Serialise a single constituent event for JSON output."""
    on_start = ev.get('on_start')
    off_end = ev.get('off_end') or ev.get('off_start')

    result = {
        'phase': ev.get('phase'),
        'on_start': _ts(on_start),
        'on_end': _ts(ev.get('on_end')),
        'off_start': _ts(ev.get('off_start')),
        'off_end': _ts(off_end),
        'on_magnitude': _val(ev.get('on_magnitude')),
        'off_magnitude': _val(ev.get('off_magnitude')),
        'duration': _val(ev.get('duration')),
        'tag': ev.get('tag'),
        'iteration': ev.get('iteration'),
        'threshold': ev.get('threshold'),
    }

    # Include per-minute power profile from segmentation (actual extraction shape)
    if device_profiles:
        on_event_id = ev.get('on_event_id')
        if on_event_id:
            # device_profiles: {run_number: {on_event_id: {timestamps, values}}}
            for run_profiles in device_profiles.values():
                profile = run_profiles.get(on_event_id)
                if profile:
                    result['power_profile'] = {
                        'timestamps': [_ts(t) for t in profile['timestamps']],
                        'values': [round(v, 1) for v in profile['values']],
                    }
                    break

    return result


def _build_summary(classified: Dict[str, List[ClassifiedSession]]) -> dict:
    """Build summary statistics for the JSON output."""
    by_type = {}
    total = 0
    for dtype, cs_list in classified.items():
        count = len(cs_list)
        total += count
        if count == 0:
            continue
        durations = []
        cycles = []
        for cs in cs_list:
            s = cs.session
            if isinstance(s, MultiPhaseSession):
                dur = (s.end - s.start).total_seconds() / 60 if s.start and s.end else 0
                cyc = sum(ps.cycle_count for ps in s.phase_sessions.values())
            else:
                dur = s.total_duration_minutes
                cyc = s.cycle_count
            durations.append(dur)
            cycles.append(cyc)

        entry = {'count': count}
        if durations:
            entry['avg_duration_min'] = round(float(np.mean(durations)), 1)
        if cycles:
            entry['avg_cycle_count'] = round(float(np.mean(cycles)), 1)
        by_type[dtype] = entry

    return {
        'total_sessions': total,
        'by_device_type': by_type,
    }


def _save_backward_compatible_json(
    flat_events: List[dict],
    house_id: str,
    threshold_schedule: List[int],
    experiment_dir: Path,
) -> None:
    """Save device_activations_{house_id}.json for backward compatibility."""
    activations_dir = experiment_dir / "device_activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    output_path = activations_dir / f"device_activations_{house_id}.json"

    flat_events.sort(key=lambda a: a.get('on_start') or '9999')

    matched_count = len(flat_events)

    output = {
        'house_id': house_id,
        'generated_at': datetime.now().isoformat(),
        'threshold_schedule': threshold_schedule,
        'total_activations': matched_count,
        'total_matched': matched_count,
        'total_unmatched_on': 0,
        'total_unmatched_off': 0,
        'activations': flat_events,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=_json_serializer)

    logger.info(f"Backward-compatible JSON saved: {matched_count} activations → {output_path}")


# ============================================================================
# Serialisation utilities
# ============================================================================

def _ts(val) -> Optional[str]:
    """Timestamp → ISO string or None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.isoformat()
    if isinstance(val, str):
        return val
    return str(val)


def _val(v) -> Any:
    """Numeric value → JSON-safe."""
    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, float):
        return round(v, 2)
    return v


def _json_serializer(obj):
    """Fallback serialiser for json.dump."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
