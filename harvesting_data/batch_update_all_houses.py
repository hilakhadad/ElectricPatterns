import pandas as pd
from fetch_and_store_house_data import update_single_house

HOUSE_LIST_CSV = "small_mishkit.csv"  # path to your house metadata file

def load_house_tokens(path=HOUSE_LIST_CSV):
    df = pd.read_csv(path)

    house_tokens = []
    for _, row in df.iterrows():
        house_id = row["ID"]
        token = str(row["Token"]).strip().lstrip("'")
        if token:
            house_tokens.append((house_id, token))

    return house_tokens

def update_all_houses():
    houses = load_house_tokens()
    for house_id, token in houses:
        try:
            update_single_house(house_id, token)
        except Exception as e:
            print(f"Error updating house {house_id}: {e}")

if __name__ == "__main__":
    update_all_houses()
