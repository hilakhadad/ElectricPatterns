"""
Shared HTML utility functions used across analysis modules.

These functions were originally duplicated in:
  - house_analysis/src/visualization/html_report.py
  - disaggregation_analysis/src/visualization/dynamic_html_report.py
  - disaggregation_analysis/src/visualization/html_report.py
  - identification_analysis/src/visualization/identification_html_report.py

Canonical versions live here; modules import from shared.html_utils.
"""


# ---------------------------------------------------------------------------
# Unified CSS constants — matches github_pages_index.html design language
# ---------------------------------------------------------------------------

# Header gradient
EP_HEADER_GRADIENT = 'linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%)'
# Background
EP_BG = '#FAFBFF'
# Text
EP_TEXT_PRIMARY = '#3D3D50'
EP_TEXT_SECONDARY = '#7D7D92'
# Cards
EP_CARD_BG = '#FFFFFF'
EP_CARD_RADIUS = '14px'
EP_CARD_SHADOW = '0 2px 12px rgba(120,100,160,0.07)'
EP_CARD_BORDER = '#E8E4F0'
# Font stack
EP_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"
# Table
EP_TABLE_HEADER_BG = '#7888A0'
EP_SECTION_BORDER_BOTTOM = '#E8E4F0'
# Hover
EP_HOVER_BG = '#F5F4FA'


def get_unified_css() -> str:
    """Return the unified CSS block matching the ElectricPatterns design language.

    Drop this into any report <style> tag to get consistent look-and-feel.
    Uses double-brace escaping so it can be embedded inside an f-string.
    """
    return f"""
        * {{{{ margin: 0; padding: 0; box-sizing: border-box; }}}}

        body {{{{
            font-family: {EP_FONT};
            background-color: {EP_BG};
            color: {EP_TEXT_PRIMARY};
            line-height: 1.6;
        }}}}

        .container {{{{ max-width: 1400px; margin: 0 auto; padding: 20px; }}}}

        header {{{{
            background: {EP_HEADER_GRADIENT};
            color: white;
            padding: 40px 30px;
            margin-bottom: 30px;
            border-radius: 16px;
            text-align: left;
        }}}}

        header h1 {{{{
            font-size: 2.2em;
            margin-bottom: 8px;
            letter-spacing: -0.3px;
            font-weight: 700;
        }}}}

        header .subtitle {{{{
            font-size: 1.05em;
            opacity: 0.92;
        }}}}

        .info-bar {{{{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}}}

        .info-item {{{{
            font-size: 0.9em;
            opacity: 0.85;
        }}}}

        .info-item strong {{{{
            opacity: 1;
        }}}}

        section {{{{
            background: {EP_CARD_BG};
            border-radius: {EP_CARD_RADIUS};
            padding: 28px;
            margin-bottom: 22px;
            box-shadow: {EP_CARD_SHADOW};
            border: 1px solid {EP_CARD_BORDER};
        }}}}

        section h2 {{{{
            color: {EP_TEXT_PRIMARY};
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid {EP_SECTION_BORDER_BOTTOM};
            font-size: 1.35em;
        }}}}

        /* Summary cards */
        .summary-grid {{{{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}}}

        .summary-card {{{{
            background: {EP_BG};
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid {EP_CARD_BORDER};
        }}}}

        .summary-number {{{{
            font-size: 2.2em;
            font-weight: bold;
            color: {EP_TEXT_PRIMARY};
        }}}}

        .summary-label {{{{
            color: {EP_TEXT_SECONDARY};
            margin-top: 5px;
            font-size: 0.9em;
        }}}}

        /* Data table */
        .data-table {{{{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}}}

        .data-table th {{{{
            background: {EP_TABLE_HEADER_BG};
            color: white;
            padding: 10px 14px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}}}

        .data-table th:first-child {{{{ border-radius: 8px 0 0 0; }}}}
        .data-table th:last-child {{{{ border-radius: 0 8px 0 0; }}}}

        .data-table td {{{{
            padding: 10px 14px;
            border-bottom: 1px solid {EP_CARD_BORDER};
        }}}}

        .data-table tr:hover {{{{
            background: {EP_HOVER_BG};
        }}}}

        .data-table tr.hidden-row,
        .data-table tr.row-hidden {{{{
            display: none;
        }}}}

        /* Badges */
        .badge {{{{
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
        }}}}

        .badge-green {{{{ background: #D8F0E0; color: #3A6A4A; }}}}
        .badge-blue {{{{ background: #D0E4F4; color: #2A5A7A; }}}}
        .badge-orange {{{{ background: #F5ECD5; color: #6A5A2A; }}}}
        .badge-red {{{{ background: #F5D8D8; color: #6A3030; }}}}
        .badge-purple {{{{ background: #E5D8F0; color: #5A3A7A; }}}}
        .badge-purple-light {{{{ background: #d4c5e2; color: #5a3d7a; }}}}
        .badge-purple-dark {{{{ background: #c9a3d4; color: #4a0e6b; }}}}

        /* House links */
        .house-link {{{{
            color: #7B9BC4;
            text-decoration: none;
            transition: color 0.2s;
        }}}}

        .house-link:hover {{{{
            color: #B488B4;
            text-decoration: underline;
        }}}}

        /* Filter bar */
        .filter-bar {{{{
            background: {EP_BG};
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 15px;
            border: 1px solid {EP_CARD_BORDER};
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
        }}}}

        .filter-bar label {{{{
            font-weight: 600;
            color: {EP_TEXT_PRIMARY};
            margin-right: 5px;
        }}}}

        /* Chart containers */
        .chart-container {{{{
            margin-bottom: 30px;
        }}}}

        .chart-content {{{{
            min-height: 400px;
        }}}}

        /* Charts grid */
        .charts-grid {{{{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}}}

        footer {{{{
            text-align: center;
            padding: 20px;
            color: {EP_TEXT_SECONDARY};
            font-size: 0.9em;
        }}}}

        @media (max-width: 768px) {{{{
            .container {{{{ padding: 10px; }}}}
            header {{{{ padding: 20px; }}}}
            section {{{{ padding: 15px; }}}}
            .data-table {{{{ font-size: 0.8em; }}}}
            .charts-grid {{{{ grid-template-columns: 1fr; }}}}
            .info-bar {{{{ flex-direction: column; gap: 5px; }}}}
        }}}}
    """


def build_glossary_section() -> str:
    """Return a collapsible glossary HTML section (collapsed by default).

    Include this in every report by appending the returned string
    before the </div> closing the .container.
    """
    return '''
    <section id="glossary-section" style="border-left: 4px solid #B488B4;">
        <h2 onclick="this.parentElement.classList.toggle('collapsed')"
            style="cursor:pointer;user-select:none;position:relative;padding-right:30px;">
            Glossary
            <span style="position:absolute;right:0;top:50%;transform:translateY(-60%);
                         font-size:0.7em;color:#C0BCD0;" class="glossary-arrow">&#x25BE;</span>
        </h2>
        <div class="section-body" id="glossary-body" style="display:none;">
            <p style="color:#7D7D92;font-size:0.88em;margin-bottom:14px;">
                Definitions for technical terms used throughout this report.
            </p>
            <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
                <thead>
                    <tr style="background:#7888A0;color:white;">
                        <th style="padding:8px 14px;text-align:left;border-radius:8px 0 0 0;width:200px;">Term</th>
                        <th style="padding:8px 14px;text-align:left;border-radius:0 8px 0 0;">Definition</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Segregation</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">Separating total household power into device-specific consumption. The pipeline detects when each device turns ON and OFF, then subtracts that power from the total.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Match</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">An ON event paired with its corresponding OFF event, forming one device activation. A match defines when a device started, how long it ran, and how much power it used.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Session</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">A continuous period of device activity consisting of one or more temporally close matches on the same phase. Matches within 30 minutes of each other are grouped into one session.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Remaining Power</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">The power left after subtracting all detected device activations from the total signal. Lower remaining means more of the consumption has been explained.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Phase (w1 / w2 / w3)</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">One of 3 independent electrical circuits in an Israeli household. Most devices use a single phase; large devices like central AC may use 2 or 3 phases simultaneously.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Iteration</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">One pass of the detection algorithm at a specific power threshold. The pipeline runs 4 iterations (2000W, 1500W, 1100W, 800W), each detecting progressively smaller devices.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Confidence</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">How certain the algorithm is about a device classification, scored 0-100. Based on how well the session characteristics match the expected device profile.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">Spike / Transient</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">A very short power event (under 3 minutes) likely caused by noise, motor starts, or brief appliance use. These are filtered out before device classification.</td></tr>
                    <tr><td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;font-weight:600;">CV (Coefficient of Variation)</td>
                        <td style="padding:8px 14px;border-bottom:1px solid #E8E4F0;">A measure of how stable the power reading is during an event (standard deviation / mean). Lower CV means more stable, consistent power draw.</td></tr>
                    <tr><td style="padding:8px 14px;font-weight:600;">Three-Phase Device</td>
                        <td style="padding:8px 14px;">A device using all 3 electrical phases simultaneously (e.g., central air conditioner, EV charger). Identified by synchronized activity across phases.</td></tr>
                </tbody>
            </table>
        </div>
    </section>
    <script>
    (function() {
        var glossary = document.getElementById('glossary-section');
        var body = document.getElementById('glossary-body');
        var h2 = glossary ? glossary.querySelector('h2') : null;
        if (h2) {
            h2.addEventListener('click', function() {
                var isHidden = body.style.display === 'none';
                body.style.display = isHidden ? 'block' : 'none';
                var arrow = h2.querySelector('.glossary-arrow');
                if (arrow) arrow.innerHTML = isHidden ? '&#x25BE;' : '&#x25B8;';
            });
        }
    })();
    </script>
    '''


def build_about_section(report_type: str) -> str:
    """Return a collapsible 'About This Report' HTML section.

    Args:
        report_type: One of 'disaggregation', 'identification', 'house_pre_analysis'.
    """
    texts = {
        'disaggregation': (
            'About This Report',
            'This report analyzes the results of the power disaggregation pipeline. '
            'The algorithm detects individual device activations (ON/OFF events) within '
            'the total household power consumption and extracts their power usage. '
            'Higher segregation rates mean more of the total consumption has been '
            'attributed to specific devices.'
        ),
        'identification': (
            'About This Report',
            'This report shows how detected device activations were classified into '
            'device types (boiler, air conditioner, EV charger, etc.). Classification '
            'is based on power patterns, duration, and phase behavior. Higher confidence '
            'means the algorithm is more certain about the device type.'
        ),
        'house_pre_analysis': (
            'About This Report',
            'This report assesses the quality and characteristics of the raw power data '
            'before running the analysis pipeline. It checks data coverage, phase balance, '
            'and power patterns to determine how suitable this house is for device detection.'
        ),
    }

    title, description = texts.get(report_type, ('About This Report', ''))
    if not description:
        return ''

    return f'''
    <section id="about-section" style="border-left: 4px solid #A8D8EA;">
        <h2 onclick="(function(el){{
                var body=el.nextElementSibling;
                var isHidden=body.style.display==='none';
                body.style.display=isHidden?'block':'none';
                var arrow=el.querySelector('.about-arrow');
                if(arrow) arrow.innerHTML=isHidden?'&#x25BE;':'&#x25B8;';
            }})(this)"
            style="cursor:pointer;user-select:none;position:relative;padding-right:30px;">
            {title}
            <span style="position:absolute;right:0;top:50%;transform:translateY(-60%);
                         font-size:0.7em;color:#C0BCD0;" class="about-arrow">&#x25B8;</span>
        </h2>
        <div style="display:none;">
            <p style="color:#5D5D72;font-size:0.92em;line-height:1.7;margin-top:4px;">
                {description}
            </p>
        </div>
    </section>
    '''


def build_upstream_metric_banner(label: str, value, suffix: str = '', color: str = '#7B9BC4') -> str:
    """Return an HTML banner displaying an upstream metric from a previous report.

    Args:
        label: e.g. 'Pre-Analysis Quality Score' or 'Segregation Effectiveness'
        value: numeric score or None
        suffix: e.g. '/100' or '%'
        color: accent color for the banner
    """
    if value is None:
        display = '<span style="color:#999;">Not available</span>'
        note = 'Run the upstream report first to populate this metric.'
    else:
        if isinstance(value, float):
            display = f'<span style="font-weight:700;font-size:1.3em;color:{color};">{value:.0f}{suffix}</span>'
        else:
            display = f'<span style="font-weight:700;font-size:1.3em;color:{color};">{value}{suffix}</span>'
        note = ''

    note_html = f'<div style="font-size:0.78em;color:#999;margin-top:4px;">{note}</div>' if note else ''
    return f'''
    <div style="display:flex;align-items:center;gap:16px;padding:12px 20px;
                background:{EP_BG};border:1px solid {EP_CARD_BORDER};border-left:4px solid {color};
                border-radius:10px;margin-bottom:16px;">
        <div style="font-size:0.88em;color:{EP_TEXT_SECONDARY};">{label}:</div>
        <div>{display}</div>
        {note_html}
    </div>
    '''


def build_quality_dist_bar(tier_counts: dict, n_houses: int) -> str:
    """Build quality distribution bar HTML -- shared visual pattern across all reports.

    Args:
        tier_counts: dict mapping tier key to count. Accepted keys:
            'excellent', 'good', 'fair', 'poor',
            'faulty_dead_phase', 'faulty_high_nan', 'faulty_both', 'unknown'
        n_houses: total number of houses (denominator for percentages)

    Returns:
        HTML string with a colored distribution bar and legend.
    """
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


def assign_tier(pre_quality) -> str:
    """Assign quality tier based on pre-analysis quality score.

    Args:
        pre_quality: numeric score (0-100), a faulty string, or None.

    Returns:
        Tier string: 'excellent', 'good', 'fair', 'poor',
        'faulty_dead_phase', 'faulty_high_nan', 'faulty_both', or 'unknown'.
    """
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        return pre_quality  # 'faulty_dead_phase', 'faulty_high_nan', or 'faulty_both'
    elif pre_quality is None:
        return 'unknown'
    elif not isinstance(pre_quality, (int, float)):
        return 'unknown'
    elif pre_quality >= 90:
        return 'excellent'
    elif pre_quality >= 75:
        return 'good'
    elif pre_quality >= 50:
        return 'fair'
    else:
        return 'poor'


def format_pre_quality(pre_quality) -> str:
    """Format pre-quality score as colored HTML.

    Args:
        pre_quality: numeric score (0-100), a faulty string, or None.

    Returns:
        HTML span string with color-coded quality indicator.
    """
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        _faulty_labels = {
            'faulty_dead_phase': ('Dead Phase', 'Phase with <2% of sisters avg'),
            'faulty_high_nan': ('High NaN', 'Phase with >=10% NaN values'),
            'faulty_both': ('Both', 'Dead phase + high NaN on other phases'),
        }
        _fl, _ft = _faulty_labels.get(pre_quality, ('Faulty', ''))
        return f'<span style="color:#6f42c1;font-weight:bold;" title="{_ft}">{_fl}</span>'
    elif pre_quality is None or not isinstance(pre_quality, (int, float)):
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
