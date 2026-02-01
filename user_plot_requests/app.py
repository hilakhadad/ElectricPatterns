"""
User Plot Requests - Flask Web Application.

Web interface for viewing segmentation results by house and time window.
"""
from flask import Flask, render_template_string, request, jsonify, send_file
from datetime import datetime
from pathlib import Path

from src.data_loader import (
    get_houses_info,
    get_house_date_range,
    load_house_data,
    load_events_data,
    filter_data_by_window
)
from src.plot_generator import (
    plot_exists,
    get_cached_plot_path,
    generate_plot,
    generate_embedded_plot
)


app = Flask(__name__)


# HTML template for the main page
MAIN_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segmentation Results Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2em;
        }
        .controls {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .form-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: flex-end;
        }
        .form-group {
            flex: 1;
            min-width: 200px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        select, input[type="date"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            background: white;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #2980b9;
        }
        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .house-info {
            background: #e8f4f8;
            padding: 10px 15px;
            border-radius: 6px;
            margin-top: 15px;
            font-size: 14px;
            color: #2c3e50;
        }
        .plot-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 400px;
        }
        .loading {
            text-align: center;
            padding: 100px 20px;
            color: #7f8c8d;
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid #ddd;
            border-top-color: #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 15px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            color: #c0392b;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 8px;
        }
        .badge-success {
            background: #2ecc71;
            color: white;
        }
        .badge-warning {
            background: #f39c12;
            color: white;
        }
        .no-plot {
            text-align: center;
            padding: 100px 20px;
            color: #95a5a6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Segmentation Results Viewer</h1>

        <div class="controls">
            <div class="form-row">
                <div class="form-group">
                    <label for="house">Select House</label>
                    <select id="house" onchange="updateHouseInfo()">
                        <option value="">-- Select a house --</option>
                        {% for house in houses %}
                        <option value="{{ house.house_id }}"
                                data-start="{{ house.start_date }}"
                                data-end="{{ house.end_date }}"
                                data-has-exp="{{ house.has_experiment }}">
                            House {{ house.house_id }}
                            {% if house.has_experiment %}(with segmentation){% endif %}
                        </option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="date">Select Date</label>
                    <input type="date" id="date" disabled>
                </div>

                <div class="form-group">
                    <label for="window">Time Window</label>
                    <select id="window">
                        <option value="day">Day (06:00 - 18:00)</option>
                        <option value="night">Night (18:00 - 06:00)</option>
                    </select>
                </div>

                <div class="form-group" style="flex: 0;">
                    <button id="generateBtn" onclick="generatePlot()" disabled>
                        Generate Plot
                    </button>
                </div>
            </div>

            <div id="houseInfo" class="house-info" style="display: none;"></div>
        </div>

        <div id="plotContainer" class="plot-container">
            <div class="no-plot">
                <p>Select a house and date to view segmentation results</p>
            </div>
        </div>
    </div>

    <script>
        function updateHouseInfo() {
            const houseSelect = document.getElementById('house');
            const dateInput = document.getElementById('date');
            const generateBtn = document.getElementById('generateBtn');
            const houseInfo = document.getElementById('houseInfo');

            const selectedOption = houseSelect.options[houseSelect.selectedIndex];

            if (!selectedOption.value) {
                dateInput.disabled = true;
                generateBtn.disabled = true;
                houseInfo.style.display = 'none';
                return;
            }

            const startDate = selectedOption.dataset.start;
            const endDate = selectedOption.dataset.end;
            const hasExp = selectedOption.dataset.hasExp === 'True';

            // Configure date input
            dateInput.min = startDate;
            dateInput.max = endDate;
            dateInput.disabled = false;
            dateInput.value = startDate;

            // Enable generate button
            generateBtn.disabled = false;

            // Show house info
            let infoHTML = `<strong>Available range:</strong> ${startDate} to ${endDate}`;
            if (hasExp) {
                infoHTML += `<span class="badge badge-success">Segmentation Available</span>`;
            } else {
                infoHTML += `<span class="badge badge-warning">Raw Data Only</span>`;
            }
            houseInfo.innerHTML = infoHTML;
            houseInfo.style.display = 'block';
        }

        function generatePlot() {
            const houseId = document.getElementById('house').value;
            const date = document.getElementById('date').value;
            const window = document.getElementById('window').value;
            const container = document.getElementById('plotContainer');

            if (!houseId || !date) {
                alert('Please select a house and date');
                return;
            }

            // Show loading
            container.innerHTML = '<div class="loading">Generating plot...</div>';

            // Open plot in new tab or iframe
            const plotUrl = `/plot/view?house_id=${houseId}&date=${date}&window_type=${window}`;
            container.innerHTML = `<iframe src="${plotUrl}" style="width:100%; height:800px; border:none;"></iframe>`;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page with house selection."""
    houses = get_houses_info()
    return render_template_string(MAIN_PAGE_TEMPLATE, houses=houses)


@app.route('/plot')
def get_plot():
    """Generate or retrieve a cached plot."""
    house_id = request.args.get('house_id')
    date_str = request.args.get('date')  # YYYY-MM-DD
    window_type = request.args.get('window_type', 'day')

    if not house_id or not date_str:
        return jsonify({'error': 'Missing house_id or date parameter'})

    try:
        # Parse date
        date = datetime.strptime(date_str, '%Y-%m-%d')

        # Check cache first
        if plot_exists(house_id, date_str, window_type):
            cache_path = get_cached_plot_path(house_id, date_str, window_type)
            with open(cache_path, 'r', encoding='utf-8') as f:
                # Return just the plot div (strip full HTML wrapper for embedding)
                html = f.read()
                # Find the plotly div content
                start = html.find('<div id="')
                if start > 0:
                    # Return the embedded version
                    pass

        # Load data
        df = load_house_data(house_id)
        if df is None:
            return jsonify({'error': f'No data found for house {house_id}'})

        # Filter to requested window
        df_filtered = filter_data_by_window(df, date, window_type)
        if df_filtered.empty:
            return jsonify({'error': f'No data for {date_str} ({window_type} window)'})

        # Generate plot (embedded version)
        plot_html = generate_embedded_plot(df_filtered, house_id, date_str, window_type)

        # Also save full version to cache
        generate_plot(df_filtered, house_id, date_str, window_type, save_to_cache=True)

        return jsonify({'html': plot_html})

    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {e}'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/plot/view')
def view_plot():
    """Return full HTML plot for iframe embedding."""
    from flask import make_response

    house_id = request.args.get('house_id')
    date_str = request.args.get('date')  # YYYY-MM-DD
    window_type = request.args.get('window_type', 'day')

    if not house_id or not date_str:
        return "<h1>Error: Missing house_id or date parameter</h1>"

    try:
        # Parse date
        date = datetime.strptime(date_str, '%Y-%m-%d')

        # Skip cache for now to regenerate with events
        # TODO: Implement versioned cache that includes events
        # if plot_exists(house_id, date_str, window_type):
        #     cache_path = get_cached_plot_path(house_id, date_str, window_type)
        #     with open(cache_path, 'r', encoding='utf-8') as f:
        #         html = f.read()
        #         response = make_response(html)
        #         response.headers['Content-Type'] = 'text/html; charset=utf-8'
        #         return response

        # Load data
        df = load_house_data(house_id)
        if df is None:
            return f"<h1>Error: No data found for house {house_id}</h1>"

        # Filter to requested window
        df_filtered = filter_data_by_window(df, date, window_type)
        if df_filtered.empty:
            return f"<h1>Error: No data for {date_str} ({window_type} window)</h1>"

        # Load events data and filter to window
        events_df = load_events_data(house_id)
        if events_df is not None and not events_df.empty:
            # Filter events to the same time window
            window_start = df_filtered['timestamp'].min()
            window_end = df_filtered['timestamp'].max()
            events_df = events_df[
                (events_df['start'] >= window_start) &
                (events_df['start'] <= window_end)
            ]

        # Generate and cache full plot
        html = generate_plot(df_filtered, house_id, date_str, window_type,
                            save_to_cache=True, events_df=events_df)
        response = make_response(html)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response

    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"


@app.route('/plot/download')
def download_plot():
    """Download the full HTML plot file."""
    house_id = request.args.get('house_id')
    date_str = request.args.get('date')
    window_type = request.args.get('window_type', 'day')

    if not house_id or not date_str:
        return jsonify({'error': 'Missing parameters'})

    cache_path = get_cached_plot_path(house_id, date_str, window_type)

    if cache_path.exists():
        return send_file(cache_path, as_attachment=True)

    return jsonify({'error': 'Plot not found. Generate it first.'})


@app.route('/api/houses')
def api_houses():
    """API endpoint for house information."""
    return jsonify(get_houses_info())


@app.route('/api/house/<house_id>/range')
def api_house_range(house_id):
    """API endpoint for house date range."""
    date_range = get_house_date_range(house_id)
    if date_range:
        return jsonify({
            'house_id': house_id,
            'start_date': date_range[0].strftime('%Y-%m-%d'),
            'end_date': date_range[1].strftime('%Y-%m-%d')
        })
    return jsonify({'error': 'House not found'})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Segmentation Results Viewer")
    print("=" * 50)
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)
