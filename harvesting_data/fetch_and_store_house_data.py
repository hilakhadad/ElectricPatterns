import os
import time
import pandas as pd
from datetime import datetime, timedelta
from requests import get

DATA_DIR = "/sise/shanigu-group/hilakese-dorins/PreprocessedData"

MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 2
MAX_DAYS_PER_REQUEST = 7  # days
NUMBER_OF_YEARS = 6

def format_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def fetch_data_from_api(token, from_time, to_time):
    all_dataframes = []
    current_start = from_time

    while current_start < to_time:
        current_end = min(current_start + MAX_DAYS_PER_REQUEST * 86400, to_time)
        url_template = (
            "http://www.energyhive.com/mobile_proxy/getHV?fromTime={}&toTime={}&"
            "aggPeriod=minute&aggFunc=sum&offset=-180&type=PWER&period=custom&token={}"
        )
        url = url_template.format(current_start, current_end, token)

        success = False
        skip_remaining_attempts = False

        for attempt in range(MAX_RETRIES):
            try:
                response = get(url)
                json_data = response.json()

                if json_data.get('status') == 'error':
                    print(f"Attempt {attempt + 1} (from {format_ts(current_start)}): API returned error")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue

                data = json_data.get("data", [])
                if not data:
                    print(f"Empty data (from {format_ts(current_start)}); skipping retries")
                    skip_remaining_attempts = True
                    break  # no point in retrying

                df = parse_api_data(data)
                all_dataframes.append(df)
                success = True
                break

            except Exception as e:
                print(f"Attempt {attempt + 1} (from {format_ts(current_start)}): Exception occurred: {e}")
                time.sleep(RETRY_DELAY_SECONDS)

        if not success and not skip_remaining_attempts:
            print(f"Failed to fetch data for range {format_ts(current_start)} to {format_ts(current_end)}")

        current_start = current_end  # advance window

    if all_dataframes:
        return pd.concat(all_dataframes).drop_duplicates(subset=["Timestamp"])
    else:
        return None


def parse_api_data(data):
    d1, d2, d3 = data[0]["data"], data[1]["data"], data[2]["data"]
    rows = []

    for i in range(1, len(d1)):
        key = list(d1[i].keys())[0]
        timestamp = datetime.fromtimestamp(int(key[:-3]))

        row = [
            timestamp,
            None if d1[i][key] == 'undef' else d1[i][key],
            None if d2[i][key] == 'undef' else d2[i][key],
            None if d3[i][key] == 'undef' else d3[i][key]
        ]
        rows.append(row)

    return pd.DataFrame(rows, columns=["timestamp", "1", "2", "3"])

def get_latest_timestamp(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        return df["timestamp"].max()
    except Exception:
        return None

def append_new_data(house_id, new_data):
    file_path = os.path.join(DATA_DIR, f"{house_id}.csv")
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, parse_dates=["timestamp"])
        existing_data = existing_data.drop(columns=["sum"], errors='ignore')
        combined = pd.concat([existing_data, new_data]).drop_duplicates(subset="timestamp")
        combined = combined.sort_values("timestamp")
    else:
        combined = new_data.sort_values("timestamp")

    combined.to_csv(file_path, index=False)
    print(f"Saved {len(combined)} rows for house {house_id}")

def update_single_house(house_id, token):
    file_path = os.path.join(DATA_DIR, f"{house_id}.csv")
    latest_timestamp = get_latest_timestamp(file_path)

    if latest_timestamp is None:
        from_time = int((datetime.now() - timedelta(days=(365*NUMBER_OF_YEARS))).timestamp())
    else:
        from_time = int((latest_timestamp + timedelta(minutes=1)).timestamp())

    to_time = int(datetime.now().timestamp())

    print(f"Fetching data for house {house_id} from {datetime.fromtimestamp(from_time)} to {datetime.fromtimestamp(to_time)}")
    data = fetch_data_from_api(token, from_time, to_time)
    if data is not None and not data.empty:
        append_new_data(house_id, data)
    else:
        print(f"No new data for house {house_id}")
