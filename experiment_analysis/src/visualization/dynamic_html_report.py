"""
HTML report generator for dynamic threshold experiments.

Generates stand-alone HTML reports per house and aggregate reports.
Follows the same patterns as html_report.py (inline CSS, Plotly CDN).

Color scheme: Green (explained), Gray (background), Orange (improvable).
No red - avoids false impression of failure.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from metrics.dynamic_report_metrics import calculate_dynamic_report_metrics
from visualization.dynamic_report_charts import (
    create_summary_boxes,
    create_power_breakdown_bar,
    create_efficiency_gauge,
    create_threshold_waterfall,
    create_remaining_analysis_chart,
    create_device_summary_table,
)

logger = logging.getLogger(__name__)


def generate_dynamic_house_report(
    experiment_dir: str,
    house_id: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate dynamic threshold HTML report for a single house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        output_path: Where to save the HTML file (optional, auto-generated if None)

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    # Calculate metrics
    metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)

    # Generate chart sections
    summary_html = create_summary_boxes(metrics)
    breakdown_html = create_power_breakdown_bar(metrics)
    efficiency_html = create_efficiency_gauge(metrics)
    waterfall_html = create_threshold_waterfall(metrics)
    remaining_html = create_remaining_analysis_chart(metrics)
    devices_html = create_device_summary_table(metrics)

    # Build HTML document
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    period = metrics.get('data_period', {})
    threshold_schedule = metrics.get('threshold_schedule', [])

    html = _build_house_html(
        house_id=house_id,
        generated_at=generated_at,
        period=period,
        threshold_schedule=threshold_schedule,
        summary_html=summary_html,
        breakdown_html=breakdown_html,
        efficiency_html=efficiency_html,
        waterfall_html=waterfall_html,
        remaining_html=remaining_html,
        devices_html=devices_html,
        metrics=metrics,
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / f"dynamic_report_{house_id}.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")
    return output_path


def generate_dynamic_aggregate_report(
    experiment_dir: str,
    house_ids: List[str],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate aggregate report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    all_metrics = []
    for house_id in house_ids:
        metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)
        all_metrics.append(metrics)

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = _build_aggregate_html(
        all_metrics=all_metrics,
        generated_at=generated_at,
        experiment_dir=str(experiment_dir),
    )

    if output_path is None:
        output_path = str(experiment_dir / "dynamic_report_aggregate.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Aggregate report saved to {output_path}")
    return output_path


def _build_house_html(
    house_id: str,
    generated_at: str,
    period: Dict,
    threshold_schedule: List[int],
    summary_html: str,
    breakdown_html: str,
    efficiency_html: str,
    waterfall_html: str,
    remaining_html: str,
    devices_html: str,
    metrics: Dict[str, Any],
) -> str:
    """Build complete HTML document for a single house."""
    th_str = ' -> '.join(f'{t}W' for t in threshold_schedule)
    period_str = f"{period.get('start', 'N/A')} to {period.get('end', 'N/A')} ({period.get('days', 0)} days)"

    # Phase detail table
    phase_detail = _build_phase_detail_table(metrics.get('phases', {}))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Threshold Report - House {house_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            opacity: 0.9;
        }}

        .info-bar {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .info-item {{
            font-size: 0.9em;
            opacity: 0.85;
        }}

        .info-item strong {{
            opacity: 1;
        }}

        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .info-bar {{ flex-direction: column; gap: 5px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dynamic Threshold Analysis - House {house_id}</h1>
            <div class="subtitle">Generated: {generated_at}</div>
            <div class="info-bar">
                <div class="info-item"><strong>Period:</strong> {period_str}</div>
                <div class="info-item"><strong>Thresholds:</strong> {th_str}</div>
            </div>
        </header>

        <section>
            <h2>Power Decomposition Summary</h2>
            {summary_html}
        </section>

        <section>
            <h2>Power Breakdown by Phase</h2>
            {breakdown_html}
            {phase_detail}
        </section>

        <section>
            <h2>Detection Efficiency by Phase</h2>
            {efficiency_html}
        </section>

        <section>
            <h2>Threshold Contribution</h2>
            {waterfall_html}
        </section>

        <div class="charts-grid">
            <section>
                <h2>Remaining Power Analysis</h2>
                {remaining_html}
            </section>

            <section>
                <h2>Device Summary</h2>
                {devices_html}
            </section>
        </div>

        <footer>
            ElectricPatterns - Dynamic Threshold Experiment Report
        </footer>
    </div>
</body>
</html>"""


def _build_phase_detail_table(phases: Dict[str, Dict]) -> str:
    """Build a detailed comparison table for all 3 phases."""
    rows = ''
    for phase in ['w1', 'w2', 'w3']:
        ph = phases.get(phase, {})
        rows += f'''
        <tr>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; font-weight: 600;">{phase}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;">{ph.get('total_kwh', 0)} kWh</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #28a745;">{ph.get('explained_kwh', 0)} kWh ({ph.get('explained_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;">{ph.get('background_kwh', 0)} kWh ({ph.get('background_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #fd7e14;">{ph.get('improvable_kwh', 0)} kWh ({ph.get('improvable_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;">{ph.get('efficiency', 0):.1f}%</td>
        </tr>
        '''

    return f'''
    <table style="width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em;">
        <thead>
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px 15px; text-align: left;">Phase</th>
                <th style="padding: 10px 15px; text-align: right;">Total</th>
                <th style="padding: 10px 15px; text-align: right; color: #28a745;">Explained</th>
                <th style="padding: 10px 15px; text-align: right; color: #6c757d;">Background</th>
                <th style="padding: 10px 15px; text-align: right; color: #fd7e14;">Improvable</th>
                <th style="padding: 10px 15px; text-align: right;">Efficiency</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    '''


def _build_aggregate_html(
    all_metrics: List[Dict[str, Any]],
    generated_at: str,
    experiment_dir: str,
) -> str:
    """Build aggregate report for multiple houses."""
    # Compute aggregate stats
    valid = [m for m in all_metrics if m.get('totals', {}).get('total_power', 0) > 0]
    n_houses = len(valid)

    if n_houses == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir, len(all_metrics))

    efficiencies = [m['totals']['efficiency'] for m in valid]
    explained_pcts = [m['totals']['explained_pct'] for m in valid]
    background_pcts = [m['totals']['background_pct'] for m in valid]
    improvable_pcts = [m['totals']['improvable_pct'] for m in valid]

    import numpy as np
    avg_eff = np.mean(efficiencies)
    avg_explained = np.mean(explained_pcts)
    avg_background = np.mean(background_pcts)
    avg_improvable = np.mean(improvable_pcts)

    # House rows
    house_rows = ''
    for m in sorted(valid, key=lambda x: x['totals']['efficiency'], reverse=True):
        hid = m['house_id']
        t = m['totals']

        eff = t.get('efficiency', 0)
        if eff >= 70:
            eff_color = '#28a745'
        elif eff >= 50:
            eff_color = '#ffc107'
        else:
            eff_color = '#fd7e14'

        house_rows += f'''
        <tr>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee;">
                <a href="dynamic_report_{hid}.html" style="color: #667eea; text-decoration: none;">{hid}</a>
            </td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;">{t.get('total_kwh', 0)} kWh</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #28a745;">{t.get('explained_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;">{t.get('background_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #fd7e14;">{t.get('improvable_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold; color: {eff_color};">{eff:.1f}%</td>
        </tr>
        '''

    # Distribution chart for efficiency
    chart_id = 'agg-efficiency-dist'
    bins = {'<30%': 0, '30-50%': 0, '50-70%': 0, '70-85%': 0, '>85%': 0}
    for eff in efficiencies:
        if eff < 30:
            bins['<30%'] += 1
        elif eff < 50:
            bins['30-50%'] += 1
        elif eff < 70:
            bins['50-70%'] += 1
        elif eff < 85:
            bins['70-85%'] += 1
        else:
            bins['>85%'] += 1

    dist_data = json.dumps({
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {'color': ['#fd7e14', '#fd7e14', '#ffc107', '#28a745', '#28a745']},
    })
    dist_layout = json.dumps({
        'title': f'Detection Efficiency Distribution<br><sub>Avg: {avg_eff:.1f}% across {n_houses} houses</sub>',
        'xaxis': {'title': 'Efficiency Range'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False,
        'height': 350,
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Threshold - Aggregate Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        header .subtitle {{ opacity: 0.9; }}
        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-number {{ font-size: 2.2em; font-weight: bold; color: #2d3748; }}
        .summary-label {{ color: #666; margin-top: 5px; font-size: 0.9em; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .data-table th {{
            background: #2d3748;
            color: white;
            padding: 12px 15px;
            text-align: left;
        }}
        .data-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
        }}
        .data-table tr:hover {{ background: #f8f9fa; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}
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
            <h1>Dynamic Threshold - Aggregate Report</h1>
            <div class="subtitle">Generated: {generated_at} | {n_houses} houses analyzed</div>
        </header>

        <section>
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-number">{n_houses}</div>
                    <div class="summary-label">Houses</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #28a745;">{avg_explained:.1f}%</div>
                    <div class="summary-label">Avg Explained</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #6c757d;">{avg_background:.1f}%</div>
                    <div class="summary-label">Avg Background</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #fd7e14;">{avg_improvable:.1f}%</div>
                    <div class="summary-label">Avg Improvable</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #667eea;">{avg_eff:.1f}%</div>
                    <div class="summary-label">Avg Efficiency</div>
                </div>
            </div>
        </section>

        <section>
            <h2>Efficiency Distribution</h2>
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{dist_data}], {dist_layout});
            </script>
        </section>

        <section>
            <h2>Per-House Results</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>House ID</th>
                        <th style="text-align: right;">Total</th>
                        <th style="text-align: right;">Explained</th>
                        <th style="text-align: right;">Background</th>
                        <th style="text-align: right;">Improvable</th>
                        <th style="text-align: right;">Efficiency</th>
                    </tr>
                </thead>
                <tbody>
                    {house_rows}
                </tbody>
            </table>
        </section>

        <footer>
            ElectricPatterns - Dynamic Threshold Aggregate Report
        </footer>
    </div>
</body>
</html>"""


def _build_empty_aggregate_html(generated_at: str, experiment_dir: str, total: int) -> str:
    """Build aggregate report when no valid data is found."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dynamic Threshold - Aggregate Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>No Valid Data</h1>
        <p style="margin-top: 15px; color: #888;">
            Attempted to analyze {total} houses but no valid summarized data was found.
        </p>
        <p style="margin-top: 10px; color: #888; font-size: 0.9em;">
            Directory: {experiment_dir}<br>
            Generated: {generated_at}
        </p>
    </div>
</body>
</html>"""
