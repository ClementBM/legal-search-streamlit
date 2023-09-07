import pandas as pd
import streamlit as st

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_integer_dtype,
    is_object_dtype,
)

FILTERABLE_COLUMNS = [
    "Status",
    "Filing Year",
    "Case Categories",
    "Jurisdictions",
    "Principal Laws",
    "Reporter Info or Case Number",
]


def filter_dataframe(df_init: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df_init.copy()
    filter_change_name = None
    if "filters" not in st.session_state:
        st.session_state.filters = {
            filter_name: [] for filter_name in FILTERABLE_COLUMNS
        }

    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        filter_containers = {
            filter_name: st.container() for filter_name in FILTERABLE_COLUMNS
        }
        for i, column in enumerate(FILTERABLE_COLUMNS):
            if f"column-{i}" not in st.session_state:
                if get_column_category(df_init, column) == "text":
                    st.session_state[f"column-{i}"] = ""
                else:
                    st.session_state[f"column-{i}"] = []
            if st.session_state[f"column-{i}"] != st.session_state.filters[column]:
                st.session_state.filters[column] = st.session_state[f"column-{i}"]
                filter_change_name = column

    column_order_names = FILTERABLE_COLUMNS.copy()
    if filter_change_name != None:
        column_order_names.remove(filter_change_name)
        column_order_names.insert(0, filter_change_name)

    for i, column in enumerate(column_order_names):
        df = apply_filter(
            df_init,
            df,
            column,
            filter_containers,
        )

    return df


def apply_filter(df_init, df, column, filter_containers):
    i = FILTERABLE_COLUMNS.index(column)

    if isinstance(df_init[column].dtype, pd.CategoricalDtype):
        options = df[column].unique()
        user_cat_input = filter_containers[column].multiselect(
            column,
            options,
            default=st.session_state[f"column-{i}"],  # [],
            key=f"column-{i}",
        )
        if user_cat_input:
            df = df[df[column].isin(user_cat_input)]

    elif is_numeric_dtype(df_init[column]):
        _min = float(df[column].min())
        _max = float(df[column].max())

        step = (_max - _min) / 100

        if is_integer_dtype(df_init[column]):
            _min = int(df[column].min())
            _max = int(df[column].max())
            step = 1

        user_num_input = filter_containers[column].slider(
            column,
            min_value=_min,
            max_value=_max,
            value=st.session_state[f"column-{i}"],  # (_min, _max),
            step=step,
            key=f"column-{i}",
        )
        if user_num_input:
            df = df[df[column].between(*user_num_input)]

    elif is_column_list(df_init[column]):
        options = get_column_list_categories(df[column])
        user_list_text_input = filter_containers[column].multiselect(
            column,
            options,
            default=st.session_state[f"column-{i}"],  # [],
            key=f"column-{i}",
        )
        if user_list_text_input:
            df = search_in_column_list(
                df=df, column=column, queries=user_list_text_input
            )

    elif is_datetime64_any_dtype(df_init[column]):
        date_min = df[column].min()
        date_max = df[column].max()

        relative_delta = (date_min, date_max)

        if pd.isnull(date_min) and not pd.isnull(date_max):
            relative_delta = (df_init[column].min(), df[column].max())
        elif pd.isnull(date_max) and not pd.isnull(date_min):
            relative_delta = (df[column].min(), df_init[column].max())
        elif pd.isnull(date_max) and pd.isnull(date_min):
            relative_delta = (df_init[column].min(), df_init[column].max())

        user_date_input = filter_containers[column].date_input(
            column,
            value=relative_delta,
            key=f"column-{i}",
        )

        if len(user_date_input) == 1:
            user_date_input = pd.to_datetime(user_date_input[0])
            df = df.loc[df[column].between(user_date_input, df_init[column].max())]
        elif len(user_date_input) == 2:
            user_date_input = tuple(map(pd.to_datetime, user_date_input))
            start_date, end_date = user_date_input
            df = df.loc[df[column].between(start_date, end_date)]
    else:
        user_text_input = filter_containers[column].text_input(
            label=column,
            key=f"column-{i}",
        )
        if user_text_input:
            df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def get_column_category(df, column):
    if isinstance(df[column].dtype, pd.CategoricalDtype):
        return "category"
    if is_numeric_dtype(df[column]):
        if is_integer_dtype(df[column]):
            return "integer"
        else:
            return "float"
    if is_column_list(df[column]):
        return "list"
    if is_datetime64_any_dtype(df[column]):
        return "datetime"
    else:
        return "text"


def is_column_list(serie):
    is_column_array = serie.apply(type).mode(0).astype(str) == "<class 'list'>"
    return is_column_array.values[0]


def get_column_list_categories(serie: pd.Series):
    unique_categories = list(
        set(serie.fillna("NA").explode().astype(str).replace("nan", "NA").to_list())
    )
    return sorted(unique_categories)


def search_in_column_list(df, column, queries):
    mask = df[column].apply(lambda x: len(set(queries) & set(x)) > 0)
    return df[mask]
