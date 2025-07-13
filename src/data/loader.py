import pandas as pd

def load_data():
    """
    Load the data from the CSV file and return a DataFrame.
    """
    # read the csv file and drop the columns that are not needed
    df = pd.read_csv("data/processed/new.csv", sep=";", index_col="timestamp")
    df = df.drop(["timeOpen", "timeClose", "timeHigh", "timeLow", "name"], axis=1)
    
    # Convert numeric columns to float
    numeric_columns = ["open", "high", "low", "close", "volume", "marketCap"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df