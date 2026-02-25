"""
HTML template functions for disaggregation analysis reports.

Contains CSS and HTML boilerplate for aggregate and single-house reports.
Extracted from html_report.py.
"""
import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import build_glossary_section, build_about_section

logger = logging.getLogger(__name__)


def _build_html_document(title: str, summary: str, table: str,
                         charts: str, house_data_json: str,
                         generated_at: str) -> str:
    """Build complete HTML document with quality-tier filtering."""
    about_html = build_about_section('disaggregation')
    glossary_html = build_glossary_section()
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #FAFBFF;
            color: #3D3D50;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
            padding: 40px 30px;
            margin-bottom: 30px;
            border-radius: 16px;
        }}

        header h1 {{
            font-size: 2.2em;
            margin-bottom: 8px;
            letter-spacing: -0.3px;
            font-weight: 700;
        }}

        header .subtitle {{
            opacity: 0.92;
            font-size: 1.05em;
        }}

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

        /* Filter bar */
        .filter-bar {{
            background: #FAFBFF;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 25px;
            box-shadow: 0 2px 12px rgba(120,100,160,0.07);
            border: 1px solid #E8E4F0;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 15px;
        }}

        .filter-bar label {{
            font-weight: bold;
            color: #3D3D50;
            margin-right: 5px;
        }}

        .filter-checkbox {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 6px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: opacity 0.2s;
            user-select: none;
        }}

        .filter-checkbox input {{
            cursor: pointer;
        }}

        .filter-checkbox.tier-excellent {{ background: #D8F0E0; color: #3A6A4A; }}
        .filter-checkbox.tier-good {{ background: #D0E4F4; color: #2A5A7A; }}
        .filter-checkbox.tier-fair {{ background: #F5ECD5; color: #6A5A2A; }}
        .filter-checkbox.tier-poor {{ background: #F5D8D8; color: #6A3030; }}
        .filter-checkbox.tier-faulty_dead_phase {{ background: #d4c5e2; color: #5a3d7a; }}
        .filter-checkbox.tier-faulty_high_nan {{ background: #E5D8F0; color: #5A3A7A; }}
        .filter-checkbox.tier-faulty_both {{ background: #c9a3d4; color: #4a0e6b; }}
        .filter-checkbox.tier-unknown {{ background: #e9ecef; color: #495057; }}

        .filter-checkbox.unchecked {{
            opacity: 0.4;
        }}

        .filter-btn {{
            padding: 5px 12px;
            border: 1px solid #E8E4F0;
            border-radius: 15px;
            background: #fff;
            cursor: pointer;
            font-size: 0.8em;
            color: #7D7D92;
            transition: background 0.2s;
        }}

        .filter-btn:hover {{
            background: #F5F4FA;
        }}

        .filter-status {{
            font-size: 0.85em;
            color: #7D7D92;
            margin-left: auto;
        }}

        /* Summary cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}

        .summary-card {{
            background: #FAFBFF;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #E8E4F0;
        }}

        .summary-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3D3D50;
        }}

        .summary-label {{
            color: #7D7D92;
            margin-top: 5px;
        }}

        .summary-alerts ul {{
            list-style: none;
            padding: 0;
        }}

        .summary-alerts li {{
            padding: 10px 15px;
            margin: 5px 0;
            background: #FAFBFF;
            border-radius: 8px;
            border-left: 4px solid #E8E4F0;
        }}

        .summary-alerts li.alert {{
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }}

        /* Data table */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        .data-table th {{
            background: #7888A0;
            color: white;
            padding: 10px 14px;
            text-align: left;
            cursor: pointer;
            user-select: none;
        }}

        .data-table th:first-child {{ border-radius: 8px 0 0 0; }}
        .data-table th:last-child {{ border-radius: 0 8px 0 0; }}

        .data-table th:hover {{
            background: #8898B0;
        }}

        .data-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid #E8E4F0;
        }}

        .data-table tr:hover {{
            background: #F5F4FA;
        }}

        .data-table tr.hidden-row {{
            display: none;
        }}

        /* Badges */
        .badge {{
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 5px;
        }}

        .badge-purple {{ background: #E5D8F0; color: #5A3A7A; }}
        .badge-green {{ background: #D8F0E0; color: #3A6A4A; }}
        .badge-blue {{ background: #D0E4F4; color: #2A5A7A; }}
        .badge-orange {{ background: #F5ECD5; color: #6A5A2A; }}
        .badge-red {{ background: #F5D8D8; color: #6A3030; }}

        /* House links in table */
        .house-link {{
            color: #7B9BC4;
            text-decoration: none;
            transition: color 0.2s;
        }}

        .house-link:hover {{
            color: #B488B4;
            text-decoration: underline;
        }}

        /* Chart containers */
        .chart-container {{
            margin-bottom: 30px;
        }}

        .chart-content {{
            min-height: 400px;
        }}

        /* Score tiers */
        .tiers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}

        .tier-card {{
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid;
        }}

        .tier-purple {{ background: #E5D8F0; border-color: #5A3A7A; }}
        .tier-green {{ background: #D8F0E0; border-color: #3A6A4A; }}
        .tier-blue {{ background: #D0E4F4; border-color: #2A5A7A; }}
        .tier-orange {{ background: #F5ECD5; border-color: #6A5A2A; }}
        .tier-red {{ background: #F5D8D8; border-color: #6A3030; }}

        .tier-count {{
            font-size: 1.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .tier-houses {{
            font-size: 0.9em;
            color: #7D7D92;
            word-break: break-word;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #7D7D92;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .data-table {{ font-size: 0.8em; }}
            .filter-bar {{ flex-direction: column; align-items: flex-start; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="subtitle">Generated: {generated_at}</div>
        </header>

        {about_html}

        <div class="filter-bar" id="filter-bar">
            <label>Filter by Pre-Quality:</label>
            <span class="filter-checkbox tier-excellent">
                <input type="checkbox" value="excellent" checked onchange="updateFilter()"> Excellent <span class="tier-count-label" id="count-excellent"></span>
            </span>
            <span class="filter-checkbox tier-good">
                <input type="checkbox" value="good" checked onchange="updateFilter()"> Good <span class="tier-count-label" id="count-good"></span>
            </span>
            <span class="filter-checkbox tier-fair">
                <input type="checkbox" value="fair" checked onchange="updateFilter()"> Fair <span class="tier-count-label" id="count-fair"></span>
            </span>
            <span class="filter-checkbox tier-poor">
                <input type="checkbox" value="poor" checked onchange="updateFilter()"> Poor <span class="tier-count-label" id="count-poor"></span>
            </span>
            <span class="filter-checkbox tier-faulty_dead_phase">
                <input type="checkbox" value="faulty_dead_phase" checked onchange="updateFilter()"> Dead Phase <span class="tier-count-label" id="count-faulty_dead_phase"></span>
            </span>
            <span class="filter-checkbox tier-faulty_high_nan">
                <input type="checkbox" value="faulty_high_nan" checked onchange="updateFilter()"> High NaN <span class="tier-count-label" id="count-faulty_high_nan"></span>
            </span>
            <span class="filter-checkbox tier-faulty_both">
                <input type="checkbox" value="faulty_both" checked onchange="updateFilter()"> Both <span class="tier-count-label" id="count-faulty_both"></span>
            </span>
            <span class="filter-checkbox tier-unknown">
                <input type="checkbox" value="unknown" checked onchange="updateFilter()"> Unknown <span class="tier-count-label" id="count-unknown"></span>
            </span>
            <span style="margin-left: 10px;">
                <button class="filter-btn" onclick="selectAll()">Select All</button>
                <button class="filter-btn" onclick="deselectAll()">Deselect All</button>
                <button class="filter-btn" onclick="allExceptFaulty()" style="font-weight: bold;">All except Faulty</button>
            </span>
            <span class="filter-status" id="filter-status"></span>
        </div>

        <section>
            <h2>Executive Summary</h2>
            {summary}
        </section>

        <section>
            <h2>House Comparison</h2>
            {table}
        </section>

        <section>
            <h2>Analysis Charts</h2>
            {charts}
        </section>

        {glossary_html}

        <footer>
            ElectricPatterns &mdash; Disaggregation Analysis Report
        </footer>
    </div>

    <script>
        // Embedded per-house data for filtering
        const houseData = {house_data_json};

        // Count houses per tier and show in filter bar
        function initFilterCounts() {{
            const tiers = ['excellent', 'good', 'fair', 'poor', 'faulty_dead_phase', 'faulty_high_nan', 'faulty_both', 'unknown'];
            tiers.forEach(function(tier) {{
                const count = houseData.filter(function(h) {{ return h.tier === tier; }}).length;
                const el = document.getElementById('count-' + tier);
                if (el) el.textContent = '(' + count + ')';
                // Hide checkbox if no houses in this tier
                if (count === 0) {{
                    var parent = el.closest('.filter-checkbox');
                    if (parent) parent.style.display = 'none';
                }}
            }});
        }}

        function getCheckedTiers() {{
            var checkboxes = document.querySelectorAll('#filter-bar input[type=checkbox]');
            var selected = [];
            checkboxes.forEach(function(cb) {{
                if (cb.checked) selected.push(cb.value);
                // Toggle visual state
                var parent = cb.parentElement;
                if (cb.checked) {{
                    parent.classList.remove('unchecked');
                }} else {{
                    parent.classList.add('unchecked');
                }}
            }});
            return selected;
        }}

        function selectAll() {{
            document.querySelectorAll('#filter-bar input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
            updateFilter();
        }}

        function deselectAll() {{
            document.querySelectorAll('#filter-bar input[type=checkbox]').forEach(function(cb) {{ cb.checked = false; }});
            updateFilter();
        }}

        function allExceptFaulty() {{
            document.querySelectorAll('#filter-bar input[type=checkbox]').forEach(function(cb) {{
                cb.checked = !cb.value.startsWith('faulty');
            }});
            updateFilter();
        }}

        function updateFilter() {{
            var selected = getCheckedTiers();
            var filtered = houseData.filter(function(h) {{ return selected.indexOf(h.tier) !== -1; }});

            // Update status
            var statusEl = document.getElementById('filter-status');
            if (statusEl) {{
                statusEl.textContent = 'Showing ' + filtered.length + ' / ' + houseData.length + ' houses';
            }}

            filterTableRows(selected);
            updateSummaryCards(filtered);
            updateCharts(filtered);
            updateSummaryTable(filtered);
        }}

        function filterTableRows(selectedTiers) {{
            var table = document.getElementById('comparison-table');
            if (!table) return;
            var rows = table.querySelectorAll('tbody tr');
            rows.forEach(function(row) {{
                var tier = row.getAttribute('data-tier');
                if (selectedTiers.indexOf(tier) !== -1) {{
                    row.classList.remove('hidden-row');
                }} else {{
                    row.classList.add('hidden-row');
                }}
            }});
        }}

        function updateSummaryCards(filtered) {{
            var n = filtered.length;

            // Houses count
            var el = document.getElementById('summary-houses-count');
            if (el) el.textContent = n;

            if (n === 0) {{
                setCardValue('summary-th-explanation', '-');
                setCardValue('summary-th-std', '');
                setCardValue('summary-overall-score', '-');
                setCardValue('summary-matching-score', '-');
                setCardValue('summary-seg-ratio', '-');
                setCardValue('summary-low-matching', '0');
                setCardValue('summary-low-seg', '0');
                setCardValue('summary-negative', '0');
                return;
            }}

            // Threshold explanation
            var thRates = filtered.map(function(h) {{ return h.th_explanation_rate || 0; }});
            var avgTh = mean(thRates);
            var stdTh = stdDev(thRates);
            setCardValue('summary-th-explanation', formatPct(avgTh));
            setCardValue('summary-th-std', '(>1300W) \u00b1' + formatPct(stdTh) + ' std');

            // Color for th explanation
            var thCard = document.getElementById('summary-th-card');
            if (thCard) {{
                var thColor = avgTh >= 0.8 ? '#28a745' : avgTh >= 0.5 ? '#ffc107' : '#dc3545';
                thCard.style.border = '2px solid ' + thColor;
                thCard.style.background = 'linear-gradient(135deg, #fff 0%, ' + thColor + '22 100%)';
                var thNum = document.getElementById('summary-th-explanation');
                if (thNum) thNum.style.color = thColor;
            }}

            // Overall score
            var overallScores = filtered.map(function(h) {{ return h.overall_score || 0; }});
            setCardValue('summary-overall-score', Math.round(mean(overallScores)));

            // Matching score
            var matchScores = filtered.map(function(h) {{ return h.matching_score || 0; }});
            setCardValue('summary-matching-score', Math.round(mean(matchScores)) + '%');

            // Segmentation ratio
            var segRatios = filtered.map(function(h) {{ return h.seg_ratio || 0; }});
            setCardValue('summary-seg-ratio', formatPct(mean(segRatios)));

            // Issues counts
            var nLowMatching = filtered.filter(function(h) {{ return h.flags && h.flags.low_matching_rate; }}).length;
            var nLowSeg = filtered.filter(function(h) {{ return h.flags && h.flags.low_segmentation; }}).length;
            var nNegative = filtered.filter(function(h) {{ return h.flags && h.flags.has_negative_values; }}).length;

            setCardValue('summary-low-matching', nLowMatching);
            setCardValue('summary-low-seg', nLowSeg);
            setCardValue('summary-negative', nNegative);
        }}

        function updateSummaryTable(filtered) {{
            var container = document.getElementById('summary-table-container');
            if (!container || filtered.length === 0) {{
                if (container) container.innerHTML = '<p style="color: #999; text-align: center;">No houses selected</p>';
                return;
            }}

            // Rebuild summary table from filtered data
            var n = filtered.length;
            var matchRates = filtered.map(function(h) {{ return h.matching_rate || 0; }});
            var segRatios = filtered.map(function(h) {{ return h.seg_ratio || 0; }});
            var minRatios = filtered.map(function(h) {{ return h.minutes_ratio || 0; }});
            var thRates = filtered.map(function(h) {{ return h.th_explanation_rate || 0; }});
            var overallScores = filtered.map(function(h) {{ return h.overall_score || 0; }});

            var totalMatches = filtered.reduce(function(s, h) {{ return s + (h.total_matches || 0); }}, 0);
            var totalMatchedMin = filtered.reduce(function(s, h) {{ return s + (h.matched_minutes || 0); }}, 0);

            // Count devices
            var nCentralAC = filtered.filter(function(h) {{ return h.has_central_ac; }}).length;
            var nRegularAC = filtered.filter(function(h) {{ return h.has_regular_ac; }}).length;
            var nBoiler = filtered.filter(function(h) {{ return h.has_boiler; }}).length;

            var html = '<table style="width:100%; border-collapse: collapse; font-size: 0.95em;">';
            html += '<thead><tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">';
            html += '<th style="padding: 10px; text-align: left;">Metric</th>';
            html += '<th style="padding: 10px; text-align: center;">Mean</th>';
            html += '<th style="padding: 10px; text-align: center;">Median</th>';
            html += '<th style="padding: 10px; text-align: center;">Std</th>';
            html += '<th style="padding: 10px; text-align: center;">Min</th>';
            html += '<th style="padding: 10px; text-align: center;">Max</th>';
            html += '</tr></thead><tbody>';

            var rows = [
                ['Matching Rate (events)', matchRates, true],
                ['Segmentation (power)', segRatios, true],
                ['Segmentation (minutes)', minRatios, true],
                ['High-Power Segregated', thRates, true],
                ['Overall Score', overallScores, false],
            ];

            rows.forEach(function(r) {{
                var label = r[0], vals = r[1], isPct = r[2];
                var avg = mean(vals);
                var med = median(vals);
                var sd = stdDev(vals);
                var mn = Math.min.apply(null, vals);
                var mx = Math.max.apply(null, vals);
                var fmt = isPct ? formatPct : function(v) {{ return v.toFixed(1); }};
                html += '<tr style="border-bottom: 1px solid #eee;">';
                html += '<td style="padding: 8px 10px; font-weight: 500;">' + label + '</td>';
                html += '<td style="padding: 8px 10px; text-align: center;">' + fmt(avg) + '</td>';
                html += '<td style="padding: 8px 10px; text-align: center;">' + fmt(med) + '</td>';
                html += '<td style="padding: 8px 10px; text-align: center;">' + fmt(sd) + '</td>';
                html += '<td style="padding: 8px 10px; text-align: center;">' + fmt(mn) + '</td>';
                html += '<td style="padding: 8px 10px; text-align: center;">' + fmt(mx) + '</td>';
                html += '</tr>';
            }});

            html += '</tbody></table>';
            html += '<div style="margin-top: 15px; font-size: 0.9em; color: #555;">';
            html += '<strong>' + n + '</strong> houses | ';
            html += '<strong>' + totalMatches + '</strong> total matches | ';
            html += '<strong>' + totalMatchedMin.toLocaleString() + '</strong> matched minutes | ';
            html += 'Devices: ' + nCentralAC + ' Central AC, ' + nRegularAC + ' Regular AC, ' + nBoiler + ' Boiler';
            html += '</div>';

            container.innerHTML = html;
        }}

        function updateCharts(filtered) {{
            if (filtered.length === 0) return;

            // 1. Matching rate distribution
            updateHistogramChart('matching-dist-chart', filtered.map(function(h) {{ return (h.matching_rate || 0) * 100; }}),
                'Matching Rate Distribution', 'Matching Rate (%)', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

            // 2. Segmentation ratio distribution
            updateHistogramChart('segmentation-dist-chart', filtered.map(function(h) {{ return (h.seg_ratio || 0) * 100; }}),
                'Segmentation Ratio Distribution', 'Segmentation Ratio (%)', [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]);

            // 3. Tag breakdown pie chart
            var tagTotals = {{}};
            filtered.forEach(function(h) {{
                var tags = h.tags || {{}};
                Object.keys(tags).forEach(function(tag) {{
                    tagTotals[tag] = (tagTotals[tag] || 0) + tags[tag];
                }});
            }});
            var tagLabels = Object.keys(tagTotals);
            var tagValues = tagLabels.map(function(k) {{ return tagTotals[k]; }});

            var tagColors = {{}};
            tagColors['NON-M'] = '#e74c3c';
            tagColors['SPIKE'] = '#3498db';
            tagColors['NOISY'] = '#f39c12';
            tagColors['PARTIAL'] = '#9b59b6';

            var pieColors = tagLabels.map(function(l) {{
                return tagColors[l] || '#95a5a6';
            }});

            var tagDiv = document.getElementById('tag-breakdown-chart');
            if (tagDiv && tagLabels.length > 0) {{
                Plotly.react('tag-breakdown-chart', [{{
                    labels: tagLabels,
                    values: tagValues,
                    type: 'pie',
                    marker: {{ colors: pieColors }},
                    textinfo: 'label+percent',
                    hovertemplate: '%{{label}}: %{{value}} matches (%{{percent}})<extra></extra>'
                }}], {{
                    title: 'Match Tag Breakdown (' + filtered.length + ' houses)',
                    height: 400,
                    margin: {{ t: 40, b: 20 }}
                }});
            }}

            // 4. Duration distribution bar chart
            var durTotals = {{ 'short': 0, 'medium': 0, 'long': 0 }};
            filtered.forEach(function(h) {{
                var dur = h.durations || {{}};
                Object.keys(dur).forEach(function(d) {{
                    durTotals[d] = (durTotals[d] || 0) + dur[d];
                }});
            }});
            var durLabels = ['short', 'medium', 'long'];
            var durDisplayLabels = ['Short (\u22642min)', 'Medium (3-24min)', 'Long (\u226525min)'];
            var durValues = durLabels.map(function(k) {{ return durTotals[k] || 0; }});
            var durColors = ['#3498db', '#2ecc71', '#e67e22'];

            var durDiv = document.getElementById('duration-dist-chart');
            if (durDiv) {{
                Plotly.react('duration-dist-chart', [{{
                    x: durDisplayLabels,
                    y: durValues,
                    type: 'bar',
                    marker: {{ color: durColors }},
                    text: durValues.map(String),
                    textposition: 'outside',
                    hovertemplate: '%{{x}}: %{{y}} matches<extra></extra>'
                }}], {{
                    title: 'Match Duration Distribution (' + filtered.length + ' houses)',
                    xaxis: {{ title: 'Duration Category' }},
                    yaxis: {{ title: 'Number of Matches' }},
                    height: 400,
                    margin: {{ t: 40, b: 60 }}
                }});
            }}

            // 5. Device detection bar chart
            var nCentralAC = filtered.filter(function(h) {{ return h.has_central_ac; }}).length;
            var nRegularAC = filtered.filter(function(h) {{ return h.has_regular_ac; }}).length;
            var nBoiler = filtered.filter(function(h) {{ return h.has_boiler; }}).length;
            var nRecurring = filtered.filter(function(h) {{ return h.has_recurring_patterns; }}).length;

            var devDiv = document.getElementById('device-detection-chart');
            if (devDiv) {{
                var devLabels = ['Central AC', 'Regular AC', 'Boiler', 'Recurring Patterns'];
                var devValues = [nCentralAC, nRegularAC, nBoiler, nRecurring];
                var devPcts = devValues.map(function(v) {{ return filtered.length > 0 ? (v / filtered.length * 100).toFixed(1) + '%' : '0%'; }});
                var devColors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6'];

                Plotly.react('device-detection-chart', [{{
                    x: devLabels,
                    y: devValues,
                    type: 'bar',
                    marker: {{ color: devColors }},
                    text: devValues.map(function(v, i) {{ return v + ' (' + devPcts[i] + ')'; }}),
                    textposition: 'outside',
                    hovertemplate: '%{{x}}: %{{y}} houses<extra></extra>'
                }}], {{
                    title: 'Device Detection (' + filtered.length + ' houses)',
                    xaxis: {{ title: '' }},
                    yaxis: {{ title: 'Number of Houses' }},
                    height: 450,
                    margin: {{ t: 40, b: 80 }}
                }});
            }}
        }}

        function updateHistogramChart(chartId, values, title, xLabel, bins) {{
            var div = document.getElementById(chartId);
            if (!div || values.length === 0) return;

            // Bin the values
            var counts = new Array(bins.length - 1).fill(0);
            var binLabels = [];
            for (var i = 0; i < bins.length - 1; i++) {{
                binLabels.push(bins[i] + '-' + bins[i + 1] + '%');
                values.forEach(function(v) {{
                    if (v >= bins[i] && (i === bins.length - 2 ? v <= bins[i + 1] : v < bins[i + 1])) {{
                        counts[i]++;
                    }}
                }});
            }}

            var colors = counts.map(function(_, i) {{
                var pct = (bins[i] + bins[i + 1]) / 2;
                if (pct >= 80) return '#28a745';
                if (pct >= 60) return '#17a2b8';
                if (pct >= 40) return '#ffc107';
                if (pct >= 20) return '#fd7e14';
                return '#dc3545';
            }});

            Plotly.react(chartId, [{{
                x: binLabels,
                y: counts,
                type: 'bar',
                marker: {{ color: colors }},
                text: counts.map(String),
                textposition: 'outside',
                hovertemplate: '%{{x}}: %{{y}} houses<extra></extra>'
            }}], {{
                title: title + ' (' + values.length + ' houses)',
                xaxis: {{ title: xLabel }},
                yaxis: {{ title: 'Number of Houses' }},
                height: 400,
                margin: {{ t: 40, b: 60 }}
            }});
        }}

        // Utility functions
        function mean(arr) {{
            if (arr.length === 0) return 0;
            return arr.reduce(function(s, v) {{ return s + v; }}, 0) / arr.length;
        }}

        function median(arr) {{
            if (arr.length === 0) return 0;
            var sorted = arr.slice().sort(function(a, b) {{ return a - b; }});
            var mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        }}

        function stdDev(arr) {{
            if (arr.length <= 1) return 0;
            var m = mean(arr);
            var variance = arr.reduce(function(s, v) {{ return s + (v - m) * (v - m); }}, 0) / arr.length;
            return Math.sqrt(variance);
        }}

        function formatPct(v) {{
            return (v * 100).toFixed(1) + '%';
        }}

        function setCardValue(id, value) {{
            var el = document.getElementById(id);
            if (el) el.textContent = value;
        }}

        // Table sorting
        function sortTable(n) {{
            var table = document.getElementById("comparison-table");
            var rows = Array.from(table.querySelectorAll('tbody tr')).filter(function(r) {{
                return !r.classList.contains('hidden-row');
            }});
            var ascending = table.getAttribute('data-sort-col') !== String(n) ||
                           table.getAttribute('data-sort-dir') !== 'asc';

            rows.sort(function(a, b) {{
                var aVal = a.cells[n].textContent.trim();
                var bVal = b.cells[n].textContent.trim();

                // Try numeric comparison
                var aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                var bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}

                // Fall back to string comparison
                return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            var tbody = table.getElementsByTagName('tbody')[0];
            // Re-append all rows (visible sorted first, then hidden)
            var hiddenRows = Array.from(table.querySelectorAll('tbody tr.hidden-row'));
            rows.forEach(function(row) {{ tbody.appendChild(row); }});
            hiddenRows.forEach(function(row) {{ tbody.appendChild(row); }});

            table.setAttribute('data-sort-col', n);
            table.setAttribute('data-sort-dir', ascending ? 'asc' : 'desc');
        }}

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            initFilterCounts();
            // Set initial status
            var statusEl = document.getElementById('filter-status');
            if (statusEl) {{
                statusEl.textContent = 'Showing ' + houseData.length + ' / ' + houseData.length + ' houses';
            }}
        }});
    </script>
</body>
</html>
"""


def _build_house_html_document(house_id: str, summary: str, iterations: str,
                                matching: str, segmentation: str, patterns: str,
                                flags: str, charts: str, generated_at: str) -> str:
    """Build complete HTML document for single house."""
    about_html = build_about_section('disaggregation')
    glossary_html = build_glossary_section()
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House {house_id} - Experiment Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #FAFBFF;
            color: #3D3D50;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
            padding: 40px 30px;
            margin-bottom: 30px;
            border-radius: 16px;
        }}

        header h1 {{
            font-size: 2.2em;
            margin-bottom: 8px;
            letter-spacing: -0.3px;
            font-weight: 700;
        }}

        header .subtitle {{
            opacity: 0.92;
            font-size: 1.05em;
        }}

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

        section h4 {{
            color: #5D5D72;
            margin-bottom: 10px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .summary-card {{
            background: #FAFBFF;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #E8E4F0;
        }}

        .summary-card.highlight {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
            border: none;
        }}

        .summary-card.highlight .summary-label {{
            color: rgba(255,255,255,0.9);
        }}

        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3D3D50;
        }}

        .summary-card.highlight .summary-number {{
            color: white;
        }}

        .summary-label {{
            color: #7D7D92;
            font-size: 0.9em;
            margin-top: 5px;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-top: 8px;
        }}

        .badge-purple {{ background: #E5D8F0; color: #5A3A7A; }}
        .badge-green {{ background: #D8F0E0; color: #3A6A4A; }}
        .badge-blue {{ background: #D0E4F4; color: #2A5A7A; }}
        .badge-orange {{ background: #F5ECD5; color: #6A5A2A; }}
        .badge-red {{ background: #F5D8D8; color: #6A3030; }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}

        .data-table.small {{
            font-size: 0.85em;
        }}

        .data-table th {{
            background: #7888A0;
            color: white;
            padding: 10px 12px;
            text-align: left;
        }}

        .data-table th:first-child {{ border-radius: 8px 0 0 0; }}
        .data-table th:last-child {{ border-radius: 0 8px 0 0; }}

        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #E8E4F0;
        }}

        .data-table tr:hover {{
            background: #F5F4FA;
        }}

        .pattern-row {{
            cursor: pointer;
        }}

        .pattern-row:hover {{
            background: #F0EEF6 !important;
        }}

        .dates-row {{
            background: #FAFBFF;
        }}

        .dates-expanded {{
            padding: 15px !important;
            font-size: 0.85em;
            color: #5D5D72;
            word-break: break-word;
        }}

        .dates-cell {{
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .two-columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metrics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #E8E4F0;
        }}

        .metric {{
            display: flex;
            gap: 8px;
        }}

        .metric-label {{
            color: #7D7D92;
        }}

        .metric-value {{
            font-weight: bold;
            color: #3D3D50;
        }}

        .flags-list {{
            list-style: none;
            padding: 0;
        }}

        .flag-item {{
            padding: 10px 15px;
            margin: 5px 0;
            background: #F5ECD5;
            border-radius: 8px;
            border-left: 4px solid #D4A843;
        }}

        .flag-icon {{
            margin-right: 8px;
        }}

        .warning {{
            padding: 15px;
            background: #F5D8D8;
            border-left: 4px solid #dc3545;
            border-radius: 8px;
            margin-bottom: 15px;
            color: #6A3030;
        }}

        .success {{
            padding: 15px;
            background: #D8F0E0;
            border-left: 4px solid #3A6A4A;
            border-radius: 8px;
            color: #3A6A4A;
        }}

        .chart-container {{
            margin-bottom: 20px;
            width: 100%;
            max-width: 500px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            justify-items: start;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #7D7D92;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .two-columns {{ grid-template-columns: 1fr; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>House {house_id}</h1>
            <div class="subtitle">Experiment Analysis Report | Generated: {generated_at}</div>
        </header>

        {about_html}

        <section>
            <h2>Summary</h2>
            {summary}
        </section>

        <section>
            <h2>Issues</h2>
            {flags}
        </section>

        <section>
            <h2>Iterations Progress</h2>
            {iterations}
        </section>

        <section>
            <h2>Matching Details</h2>
            {matching}
        </section>

        <section>
            <h2>Segmentation Details</h2>
            {segmentation}
        </section>

        <section>
            <h2>Event Patterns</h2>
            {patterns}
        </section>

        <section>
            <h2>Charts</h2>
            <div class="charts-grid">
                {charts}
            </div>
        </section>

        {glossary_html}

        <footer>
            ElectricPatterns &mdash; House Disaggregation Analysis Report
        </footer>
    </div>
</body>
</html>
"""
