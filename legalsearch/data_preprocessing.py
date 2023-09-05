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


def load_global_cases(csv_path):
    global_cases_df = pd.read_csv(csv_path)

    global_cases_df["Jurisdictions"] = global_cases_df["Jurisdictions"].apply(
        preprocess_fields
    )
    global_cases_df["Case Categories"] = global_cases_df["Case Categories"].apply(
        preprocess_fields
    )
    global_cases_df["Principal Laws"] = global_cases_df["Principal Laws"].apply(
        preprocess_fields
    )
    global_cases_df["Filing Year"] = pd.to_datetime(
        global_cases_df["Filing Year"], format="%Y"
    )

    global_cases_df["columns_concatenations"] = (
        global_cases_df["Title"] + "\n" + global_cases_df["Summary"]
    )

    return global_cases_df
