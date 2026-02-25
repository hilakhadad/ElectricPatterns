"""
HTML template for house analysis aggregate report.

Contains CSS and HTML boilerplate. Extracted from html_report.py.
"""
import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import build_glossary_section, build_about_section

logger = logging.getLogger(__name__)


def _build_html_document(title: str, summary: str, filter_bar: str,
                         table: str, charts: str, quality_tiers: str,
                         generated_at: str, wave: str = '') -> str:
    """Build complete HTML document."""
    about_html = build_about_section('house_pre_analysis')
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

        /* Summary cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            color: #7B9BC4;
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

        /* Filter bar */
        .filter-bar {{
            background: #FAFBFF;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid #E8E4F0;
        }}

        .filter-group {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .filter-group:last-child {{
            margin-bottom: 0;
        }}

        .filter-group-label {{
            font-weight: 600;
            font-size: 0.85em;
            color: #3D3D50;
            min-width: 110px;
        }}

        .filter-checkbox {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 16px;
            border: 2px solid #ddd;
            font-size: 0.82em;
            cursor: pointer;
            transition: all 0.15s;
            user-select: none;
            background: white;
        }}

        .filter-checkbox:hover {{
            background: #f0f0f0;
        }}

        .filter-checkbox input {{
            display: none;
        }}

        .filter-checkbox input:not(:checked) + .filter-dot {{
            opacity: 0.3;
        }}

        .filter-checkbox input:not(:checked) ~ .filter-count {{
            opacity: 0.4;
        }}

        .filter-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        .filter-count {{
            color: #888;
            font-size: 0.9em;
        }}

        /* Table wrapper for horizontal scroll */
        .table-wrapper {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
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
            white-space: nowrap;
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

        .data-table tr[data-excluded="true"] {{
            opacity: 0.35;
        }}

        .data-table tr[data-excluded="true"] .house-link {{
            text-decoration: line-through;
        }}

        .data-table tr.row-hidden {{
            display: none;
        }}

        .data-table tr {{
            cursor: pointer;
        }}

        .house-link {{
            color: #7B9BC4;
            text-decoration: none;
        }}

        .house-link:hover {{
            color: #B488B4;
            text-decoration: underline;
        }}

        /* Badges */
        .badge {{
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 5px;
        }}

        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .badge-orange {{ background: #fff3cd; color: #856404; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}
        .badge-purple {{ background: #e2d5f1; color: #6f42c1; }}
        .badge-purple-light {{ background: #d4c5e2; color: #5a3d7a; }}
        .badge-purple-dark {{ background: #c9a3d4; color: #4a0e6b; }}

        /* Issue tags in table */
        .issues-cell {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .issue-tag {{
            display: inline-block;
            padding: 2px 6px;
            margin: 1px;
            background: #fee2e2;
            color: #991b1b;
            border-radius: 4px;
            font-size: 0.7em;
            white-space: nowrap;
        }}
        .issue-more {{
            color: #666;
            font-size: 0.8em;
            font-style: italic;
        }}
        .no-issues {{
            color: #28a745;
            font-size: 0.85em;
        }}

        /* Chart containers */
        .chart-container {{
            margin-bottom: 30px;
            max-width: 100%;
            overflow-x: auto;
        }}

        .chart-content {{
            min-height: 400px;
        }}

        /* Quality tiers - score boxes */
        .tiers-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}

        .tier-score-card {{
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid;
        }}

        .tier-score-card .tier-header {{
            margin-bottom: 12px;
        }}

        .tier-score-card h4 {{
            font-size: 0.95em;
            margin: 0;
        }}

        .tier-score-card .tier-count {{
            font-size: 0.82em;
            color: #666;
            margin-top: 2px;
        }}

        .tier-purple {{ background: #f3edf7; border-color: #6f42c1; }}
        .tier-purple-light {{ background: #ece4f0; border-color: #5a3d7a; }}
        .tier-purple-dark {{ background: #e0d0e8; border-color: #4a0e6b; }}
        .tier-green {{ background: #e8f5e9; border-color: #28a745; }}
        .tier-blue {{ background: #e3f2fd; border-color: #007bff; }}
        .tier-orange {{ background: #fff8e1; border-color: #ffc107; }}
        .tier-red {{ background: #fce4ec; border-color: #dc3545; }}

        /* Mini bars for score components */
        .mini-bars {{
            margin-bottom: 10px;
        }}

        .mini-bar-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}

        .mini-bar-label {{
            font-size: 0.75em;
            color: #555;
            min-width: 80px;
            text-align: right;
        }}

        .mini-bar-container {{
            flex: 1;
            height: 12px;
            background: rgba(0,0,0,0.08);
            border-radius: 6px;
            overflow: hidden;
        }}

        .mini-bar {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s;
        }}

        .mini-bar-value {{
            font-size: 0.72em;
            color: #666;
            min-width: 45px;
        }}

        .tier-houses {{
            font-size: 0.75em;
            color: #888;
            word-break: break-word;
            border-top: 1px solid rgba(0,0,0,0.08);
            padding-top: 8px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}

        details.ep-collapsible {{
            background: #FFFFFF; border-radius: 14px; margin-bottom: 22px;
            box-shadow: 0 2px 12px rgba(120,100,160,0.07); border: 1px solid #E8E4F0;
        }}
        details.ep-collapsible > summary {{
            cursor: pointer; padding: 20px 28px; font-size: 1.3em; font-weight: 700;
            color: #3D3D50; list-style: none; user-select: none;
        }}
        details.ep-collapsible > summary::-webkit-details-marker {{ display: none; }}
        details.ep-collapsible > summary::before {{
            content: '\\25B8'; display: inline-block; margin-right: 10px; transition: transform 0.2s;
        }}
        details.ep-collapsible[open] > summary::before {{ transform: rotate(90deg); }}
        details.ep-collapsible[open] > summary {{
            padding-bottom: 16px; border-bottom: 2px solid #E8E4F0;
        }}
        details.ep-collapsible > .collapsible-body {{ padding: 20px 28px 28px; }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .data-table {{ font-size: 0.8em; }}
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

        <section>
            <h2>Executive Summary</h2>
            {summary}
        </section>

        <details class="ep-collapsible">
            <summary>Wave Behavior</summary>
            <div class="collapsible-body">{wave}</div>
        </details>

        <details class="ep-collapsible">
            <summary>Quality Score Breakdown</summary>
            <div class="collapsible-body">{quality_tiers}</div>
        </details>

        <section>
            <h2>House Comparison</h2>
            {filter_bar}
            {table}
        </section>

        <details class="ep-collapsible">
            <summary>Analysis Charts</summary>
            <div class="collapsible-body">{charts}</div>
        </details>

        {glossary_html}

        <footer>
            ElectricPatterns &mdash; House Pre-Analysis Report
        </footer>
    </div>

    <script>
        function sortTable(n) {{
            var table = document.getElementById("comparison-table");
            var rows = Array.from(table.tBodies[0].rows);
            var ascending = table.getAttribute('data-sort-col') !== String(n) ||
                           table.getAttribute('data-sort-dir') !== 'asc';

            rows.sort(function(a, b) {{
                var aVal = a.cells[n].textContent.trim();
                var bVal = b.cells[n].textContent.trim();

                var aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                var bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}

                return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            var tbody = table.tBodies[0];
            rows.forEach(function(row) {{ tbody.appendChild(row); }});

            table.setAttribute('data-sort-col', n);
            table.setAttribute('data-sort-dir', ascending ? 'asc' : 'desc');
        }}

        function updateFilter() {{
            // Get checked tiers
            var checkedTiers = [];
            document.querySelectorAll('[data-filter-tier]').forEach(function(cb) {{
                if (cb.checked) checkedTiers.push(cb.getAttribute('data-filter-tier'));
            }});

            // Get checked continuity
            var checkedCont = [];
            document.querySelectorAll('[data-filter-continuity]').forEach(function(cb) {{
                if (cb.checked) checkedCont.push(cb.getAttribute('data-filter-continuity'));
            }});

            // Get checked wave classifications
            var checkedWave = [];
            document.querySelectorAll('[data-filter-wave]').forEach(function(cb) {{
                if (cb.checked) checkedWave.push(cb.getAttribute('data-filter-wave'));
            }});

            // Show/hide rows
            var rows = document.querySelectorAll('#comparison-table tbody tr');
            rows.forEach(function(row) {{
                var tier = row.getAttribute('data-tier');
                var cont = row.getAttribute('data-continuity');
                var wave = row.getAttribute('data-wave');
                var tierMatch = checkedTiers.length === 0 || checkedTiers.indexOf(tier) !== -1;
                var contMatch = checkedCont.length === 0 || checkedCont.indexOf(cont) !== -1;
                var waveMatch = checkedWave.length === 0 || checkedWave.indexOf(wave) !== -1;

                if (tierMatch && contMatch && waveMatch) {{
                    row.classList.remove('row-hidden');
                }} else {{
                    row.classList.add('row-hidden');
                }}
            }});

            updateSummaryCards();
        }}

        function toggleExcludeRow(event, row) {{
            // Don't toggle if clicking a link
            if (event.target.tagName === 'A' || event.target.closest('a')) return;

            var excluded = row.getAttribute('data-excluded') === 'true';
            row.setAttribute('data-excluded', excluded ? 'false' : 'true');
            updateSummaryCards();
        }}

        function updateSummaryCards() {{
            var rows = document.querySelectorAll('#comparison-table tbody tr');
            var totalHouses = 0;
            var totalScore = 0;
            var totalCoverage = 0;
            var totalDays = 0;

            rows.forEach(function(row) {{
                if (row.classList.contains('row-hidden')) return;
                if (row.getAttribute('data-excluded') === 'true') return;

                totalHouses++;
                totalScore += parseFloat(row.getAttribute('data-score') || 0);
                totalCoverage += parseFloat(row.getAttribute('data-coverage') || 0);
                totalDays += parseInt(row.getAttribute('data-days') || 0);
            }});

            var el;
            el = document.getElementById('summary-total');
            if (el) el.textContent = totalHouses;

            el = document.getElementById('summary-avg-score');
            if (el) el.textContent = totalHouses > 0 ? Math.round(totalScore / totalHouses) : 0;

            el = document.getElementById('summary-avg-coverage');
            if (el) el.textContent = totalHouses > 0
                ? (totalCoverage / totalHouses * 100).toFixed(1) + '%'
                : '0.0%';

            el = document.getElementById('summary-total-days');
            if (el) el.textContent = totalDays.toLocaleString();
        }}
    </script>
</body>
</html>
"""
