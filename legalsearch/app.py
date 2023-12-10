from datetime import date
import numpy as np
import streamlit as st

from legalsearch.data_preprocessing import load_all_cases
from legalsearch.filtering import filter_dataframe
from legalsearch.indexing import vector_index, bm25_search
from legalsearch.models import AggregatedCaseFields
import typing
from langchain.schema.document import Document


if "current_selection" not in st.session_state:
    st.session_state["current_selection"] = None


def load_data(query: str):
    all_cases_df = load_all_cases()

    total_count = len(all_cases_df)

    if query.strip() == "":
        return all_cases_df, total_count

    results: typing.List[Document] = vector_index.search(
        query, search_type="similarity"
    )
    results_ids = [result.metadata["row"] for result in results]

    bm25_result_ids = bm25_search(query, top_k=20)

    return (
        all_cases_df.iloc[list(set(results_ids).union(bm25_result_ids))],
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
    AggregatedCaseFields.PERMALINK: st.column_config.LinkColumn(
        label=AggregatedCaseFields.PERMALINK,
        width="medium",
        disabled=True,
    ),
    AggregatedCaseFields.FILING_YEAR: st.column_config.DateColumn(
        label=AggregatedCaseFields.FILING_YEAR,
        min_value=date(1900, 1, 1),
        format="YYYY",
        help="",
        width="small",
        disabled=True,
    ),
    AggregatedCaseFields.JURISDICTIONS: st.column_config.ListColumn(
        label=AggregatedCaseFields.JURISDICTIONS,
        help="Jurisdictions",
        width="small",
    ),
    AggregatedCaseFields.CATEGORIES: st.column_config.ListColumn(
        label=AggregatedCaseFields.CATEGORIES,
        help="Case Categories",
        width="small",
    ),
    AggregatedCaseFields.PRINCIPAL_LAWS: st.column_config.ListColumn(
        label=AggregatedCaseFields.PRINCIPAL_LAWS,
        help="Principal Laws",
        width="small",
    ),
    AggregatedCaseFields.TITLE: st.column_config.TextColumn(
        label=AggregatedCaseFields.TITLE,
        width="large",
        disabled=True,
    ),
    AggregatedCaseFields.SUMMARY: st.column_config.TextColumn(
        label=AggregatedCaseFields.SUMMARY,
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
            AggregatedCaseFields.TITLE,
            AggregatedCaseFields.FILING_YEAR,
            AggregatedCaseFields.JURISDICTIONS,
            AggregatedCaseFields.CATEGORIES,
            AggregatedCaseFields.PRINCIPAL_LAWS,
        ],
        column_config=df_config,
        hide_index=True,
        key="case_df",
    )

    if sum(df.Select) > 0:
        right_column.markdown("**Title**")
        right_column.write(
            f"<a href='{df[df.Select][AggregatedCaseFields.PERMALINK].values[0]}'>{df[df.Select][AggregatedCaseFields.TITLE].values[0]}</a>",
            unsafe_allow_html=True,
        )

        right_column.markdown("**Summary**")
        right_column.write(df[df.Select][AggregatedCaseFields.SUMMARY].values[0])

        right_column.markdown("**Jurisdictions**")
        for jurisdiction in df[df.Select][AggregatedCaseFields.JURISDICTIONS].values[0]:
            right_column.write(jurisdiction)

        right_column.markdown(f"**Case Categories**")
        for category in df[df.Select][AggregatedCaseFields.CATEGORIES].values[0]:
            right_column.write(category)

        right_column.markdown("**Principal Laws**")
        for principal in df[df.Select][AggregatedCaseFields.PRINCIPAL_LAWS].values[0]:
            right_column.write(principal)

        # right_column.markdown("**Reporter Info or Case Number**")
        # right_column.write(df[df.Select]["Reporter Info or Case Number"].values[0])

    modification_container.markdown(f"*Result count: {len(df)}/{total_count}*")
