import pandas as pd


def preprocess_fields(value):
    if isinstance(value, float):
        return []
    elif value:
        value = value.replace("&amp;", "&")
        values = value.split("|")
        return values
    else:
        return []


def make_clickable(link):
    # https://stackoverflow.com/questions/71641666/hyperlink-in-streamlit-dataframe
    # https://github.com/streamlit/streamlit/issues/983
    text = link.split("=")[0]
    return f'<a target="_blank" href="{link}">{text}</a>'


def load_global_cases(csv_path):
    df = pd.read_csv(csv_path)
    df.index += 1

    df["Jurisdictions"] = df["Jurisdictions"].apply(preprocess_fields)
    df["Case Categories"] = df["Case Categories"].apply(preprocess_fields)
    df["Principal Laws"] = df["Principal Laws"].apply(preprocess_fields)
    df["Filing Year"] = pd.to_datetime(df["Filing Year"], format="%Y")

    df["columns_concatenations"] = (
        df["Title"] + "\n" + df["Summary"] + "\n" + df["Core Object"]
    )

    df["Status"] = df["Status"].astype("category")

    return df
