"""
Match tag construction.

Builds human-readable tags describing match quality and duration for
matched ON/OFF event pairs.

Tag format: [NOISY-|PARTIAL-]{magnitude_quality}-{duration}[-CORRECTED][-TAIL]
"""


def get_magnitude_quality_tag(on_magnitude: float, off_magnitude: float) -> str:
    """
    Get magnitude quality tag based on ON/OFF magnitude difference.

    Tags:
    - EXACT: < 50W difference
    - CLOSE: 50-100W difference
    - APPROX: 100-200W difference
    - LOOSE: 200-350W difference
    """
    diff = abs(abs(on_magnitude) - abs(off_magnitude))
    if diff < 50:
        return "EXACT"
    elif diff < 100:
        return "CLOSE"
    elif diff < 200:
        return "APPROX"
    else:
        return "LOOSE"


def get_duration_tag(duration_minutes: float) -> str:
    """
    Get duration tag based on event duration in minutes.

    Tags:
    - SPIKE: <= 2 minutes
    - QUICK: < 5 minutes (microwave, kettle)
    - MEDIUM: 5-30 minutes (washing machine, dishwasher)
    - EXTENDED: > 30 minutes (water heater, AC)
    """
    if duration_minutes <= 2:
        return "SPIKE"
    elif duration_minutes < 5:
        return "QUICK"
    elif duration_minutes <= 30:
        return "MEDIUM"
    else:
        return "EXTENDED"


def build_match_tag(on_magnitude: float, off_magnitude: float, duration_minutes: float,
                    is_noisy: bool = False, is_partial: bool = False, is_corrected: bool = False,
                    is_tail_extended: bool = False) -> str:
    """
    Build a complete match tag combining magnitude quality and duration.

    Format: [NOISY-|PARTIAL-]{magnitude_quality}-{duration}[-CORRECTED]

    Examples:
    - EXACT-SPIKE
    - CLOSE-QUICK
    - NOISY-APPROX-MEDIUM
    - PARTIAL-EXTENDED
    - EXACT-QUICK-CORRECTED
    """
    parts = []

    # Prefix for special match types
    if is_noisy:
        parts.append("NOISY")
    elif is_partial:
        parts.append("PARTIAL")

    # Magnitude quality (skip for partial since magnitudes differ significantly)
    if not is_partial:
        parts.append(get_magnitude_quality_tag(on_magnitude, off_magnitude))

    # Duration
    parts.append(get_duration_tag(duration_minutes))

    # Suffix for corrected
    if is_corrected:
        parts.append("CORRECTED")

    # Suffix for tail-extended events
    if is_tail_extended:
        parts.append("TAIL")

    return "-".join(parts)
