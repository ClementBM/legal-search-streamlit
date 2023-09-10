from datetime import date
import numpy as np
import streamlit as st

from artefacts import CLIMATE_CASES_CSV
from legalsearch.data_preprocessing import load_global_cases
from legalsearch.filtering import filter_dataframe
from legalsearch.load_index import load
import typing
from langchain.schema.document import Document

# streamlit run legalsearch/app.py [-- script args]

# https://download.pytorch.org/whl/torch/
# https://pytorch.org/cppdocs/installing.html

# Sqlite3
# Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
# https://www.sqlite.org/chronology.html
# https://docs.trychroma.com/troubleshooting#sqlite

index = load()


if "current_selection" not in st.session_state:
    st.session_state["current_selection"] = None


def load_data(query: str):
    global_cases_df = load_global_cases(CLIMATE_CASES_CSV)
    total_count = len(global_cases_df)

    if query.strip() == "":
        return global_cases_df, total_count

    results: typing.List[Document] = index.search(query, search_type="similarity")
    results_ids = [result.metadata["row"] for result in results]

    keywords_search_idx = set(
        global_cases_df.index[
            global_cases_df["columns_concatenations"].str.contains(
                pat=query, na=False, case=False
            )
        ]
    )

    return (
        global_cases_df.iloc[list(set(results_ids).union(keywords_search_idx))],
        total_count,
    )


st.set_page_config(
    layout="wide",
    page_icon="⚖️",
    menu_items={
        "About": "This is just a demo. Please consider it as toy search engine.\nTo get the complete database: https://climate.law.columbia.edu/",
    },
)

with st.sidebar:
    search_query = st.text_input(
        "**Search query**",
        placeholder="Looking for a specific climate case?",
        key="query",
    )

df_config: dict = {
    "Case Permalink": st.column_config.LinkColumn(
        "Case Permalink",
        width="medium",
        disabled=True,
    ),
    "Filing Year": st.column_config.DateColumn(
        "Filing Year",
        min_value=date(1900, 1, 1),
        format="YYYY",
        help="",
        width="small",
        disabled=True,
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
        disabled=True,
    ),
    "Summary": st.column_config.TextColumn(
        "Summary",
        width="small",
        disabled=True,
    ),
    "Select": st.column_config.CheckboxColumn(
        required=True,
        width="small",
    ),
}


with st.spinner(text="In progress"):
    data, total_count = load_data(search_query)

    with st.sidebar:
        data, modification_container = filter_dataframe(data)

    if "case_df" in st.session_state:
        selected_rows = {
            k: v["Select"]
            for k, v in st.session_state["case_df"]["edited_rows"].items()
            if v["Select"] == True
        }

        if len(selected_rows) == 0:
            st.session_state["current_selection"] = None

        if len(selected_rows) > 0:
            st.session_state["current_selection"] = list(selected_rows.keys())[0]

    current_id = st.session_state["current_selection"]

    left_column, right_column = st.columns([2, 0.01])
    if current_id != None:
        left_column, right_column = st.columns([2, 1])

    selected_mask = np.zeros(len(data), dtype=bool)

    if current_id != None:
        selected_mask[current_id] = True

    data.insert(0, "Select", selected_mask)

    df = left_column.data_editor(
        data,
        height=650,
        use_container_width=True,
        column_order=[
            "Select",
            "Title",
            "Filing Year",
            "Jurisdictions",
            "Case Categories",
            "Principal Laws",
        ],
        column_config=df_config,
        hide_index=True,
        key="case_df",
    )

    if sum(df.Select) > 0:
        right_column.markdown("**Title**")
        right_column.write(
            f"<a href='{df[df.Select]['Case Permalink'].values[0]}'>{df[df.Select]['Title'].values[0]}</a>",
            unsafe_allow_html=True,
        )

        right_column.markdown("**Summary**")
        right_column.write(df[df.Select]["Summary"].values[0])

        right_column.markdown("**Jurisdictions**")
        for jurisdiction in df[df.Select]["Jurisdictions"].values[0]:
            right_column.write(jurisdiction)

        right_column.markdown(f"**Case Categories**")
        for category in df[df.Select]["Case Categories"].values[0]:
            right_column.write(category)

        right_column.markdown("**Principal Laws**")
        for principal in df[df.Select]["Principal Laws"].values[0]:
            right_column.write(principal)

        right_column.markdown("**Reporter Info or Case Number**")
        right_column.write(df[df.Select]["Reporter Info or Case Number"].values[0])

    modification_container.markdown(f"*Result count: {len(df)}/{total_count}*")
