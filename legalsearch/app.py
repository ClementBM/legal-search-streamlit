from datetime import date
import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import numpy as np

from artefacts import CLIMATE_CASES_CSV
from data_preprocessing import preprocess_fields, load_global_cases

# poetry export -f requirements.txt --output requirements.txt --without-hashes
# streamlit run legalsearch/app.py [-- script args]


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def load_data(query: str):
    global_cases_df = load_global_cases(CLIMATE_CASES_CSV)

    if query.strip() == "":
        return global_cases_df

    return global_cases_df[
        global_cases_df["columns_concatenations"].str.contains(
            query, na=False, case=False
        )
    ].style.highlight_max(axis=0)


st.title("Search Climate Cases")

search_query = st.text_input("", placeholder="looking for..", key="query")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")


df_config: dict = {
    "Case Permalink": st.column_config.LinkColumn(
        "Case Permalink",
        width="medium",
    ),
    "Filing Year": st.column_config.DateColumn(
        "Filing Year",
        min_value=date(1900, 1, 1),
        format="YYYY",
        help="",
        width="small",
    ),
    "Jurisdictions": st.column_config.ListColumn(
        "Jurisdictions",
        help="Jurisdictions",
        width="small",
    ),
    "Case Categories": st.column_config.ListColumn(
        "Case Categories",
        help="Case Categories",
        width="small",
    ),
    "Principal Laws": st.column_config.ListColumn(
        "Principal Laws",
        help="Principal Laws",
        width="small",
    ),
    "Title": st.column_config.TextColumn(
        "Title",
        width="large",
    ),
}

data = load_data(search_query)

case_dataframe = st.dataframe(
    data,
    use_container_width=True,
    column_order=[
        "Title",
        "Filing Year",
        "Jurisdictions",
        "Case Categories",
        "Principal Laws",
        "Summary",
        "Case Permalink",
    ],
    column_config=df_config,
)

# st.session_state.query

# Notify the reader that the data was successfully loaded.
data_load_state.text("")
