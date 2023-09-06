from datetime import date
import streamlit as st

from artefacts import CLIMATE_CASES_CSV
from legalsearch.data_preprocessing import load_global_cases
from legalsearch.load_index import load
import typing
from langchain.schema.document import Document

# streamlit run legalsearch/app.py [-- script args]
# https://download.pytorch.org/whl/torch/
# https://pytorch.org/cppdocs/installing.html

index = load()


def load_data(query: str):
    global_cases_df = load_global_cases(CLIMATE_CASES_CSV)

    if query.strip() == "":
        return global_cases_df

    results: typing.List[Document] = index.search(query, search_type="similarity")
    results_ids = [result.metadata["row"] for result in results]

    keywords_search_idx = set(
        global_cases_df.index[
            global_cases_df["columns_concatenations"].str.contains(
                pat=query, na=False, case=False
            )
        ]
    )

    return global_cases_df.iloc[list(set(results_ids).union(keywords_search_idx))]


st.set_page_config(layout="wide")

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
    height=650,
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
