import pandas as pd
import ast

from artefacts import ALL_CASES_CSV, RAW_GLOBAL_CASES_CSV, RAW_US_CASES_CSV
from legalsearch.models import AggregatedCaseFields, UsCaseFields, GlobalCaseFields


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

    df[AggregatedCaseFields.JURISDICTIONS] = df[GlobalCaseFields.JURISDICTIONS].apply(
        preprocess_fields
    )
    df[AggregatedCaseFields.CATEGORIES] = df[GlobalCaseFields.CATEGORIES].apply(
        preprocess_fields
    )
    df[AggregatedCaseFields.PRINCIPAL_LAWS] = df[GlobalCaseFields.PRINCIPAL_LAWS].apply(
        preprocess_fields
    )
    df[AggregatedCaseFields.FILING_YEAR] = pd.to_datetime(
        df[GlobalCaseFields.FILING_YEAR], format="ISO8601"
    )

    df[GlobalCaseFields.CORE_OBJECT] = (
        df[GlobalCaseFields.CORE_OBJECT].astype(str).replace("nan", "")
    )

    df[AggregatedCaseFields.SEARCHABLE_CONTENT] = (
        df[GlobalCaseFields.TITLE]
        + "\n"
        + df[GlobalCaseFields.SUMMARY]
        + "\n"
        + df[GlobalCaseFields.CORE_OBJECT]
    )

    df[AggregatedCaseFields.STATUS] = df[GlobalCaseFields.STATUS].astype("category")
    df[AggregatedCaseFields.TITLE] = df[GlobalCaseFields.TITLE]
    df[AggregatedCaseFields.PERMALINK] = df[GlobalCaseFields.PERMALINK]

    return df[AggregatedCaseFields.COLUMNS]


def load_us_cases(csv_path):
    df = pd.read_csv(csv_path)
    df.index += 1

    df[AggregatedCaseFields.CATEGORIES] = df[UsCaseFields.CATEGORIES].apply(
        preprocess_fields
    )
    df[AggregatedCaseFields.PRINCIPAL_LAWS] = df[UsCaseFields.PRINCIPAL_LAWS].apply(
        preprocess_fields
    )
    df[AggregatedCaseFields.FILING_YEAR] = pd.to_datetime(
        df[UsCaseFields.FILING_YEAR], format="ISO8601"
    )

    df[AggregatedCaseFields.SEARCHABLE_CONTENT] = (
        df[UsCaseFields.TITLE] + "\n" + df[UsCaseFields.SUMMARY]
    )

    df[AggregatedCaseFields.STATUS] = ""
    df[AggregatedCaseFields.STATUS] = df[AggregatedCaseFields.STATUS].astype("category")

    df[AggregatedCaseFields.SUMMARY] = df[UsCaseFields.SUMMARY]
    df[AggregatedCaseFields.TITLE] = df[UsCaseFields.TITLE]
    df[AggregatedCaseFields.PERMALINK] = ""
    df[AggregatedCaseFields.JURISDICTIONS] = [[]] * len(df)

    return df[AggregatedCaseFields.COLUMNS]


def load_all_cases():
    if not ALL_CASES_CSV.exists():
        global_cases_df = load_global_cases(RAW_GLOBAL_CASES_CSV)
        us_cases_df = load_us_cases(RAW_US_CASES_CSV)

        all_cases_df = pd.concat([global_cases_df, us_cases_df])
        all_cases_df.to_csv(ALL_CASES_CSV, index=False)

    df = pd.read_csv(ALL_CASES_CSV)
    df[AggregatedCaseFields.STATUS] = df[AggregatedCaseFields.STATUS].astype("category")
    df[AggregatedCaseFields.FILING_YEAR] = pd.to_datetime(
        df[AggregatedCaseFields.FILING_YEAR], format="ISO8601"
    )
    df[AggregatedCaseFields.CATEGORIES] = df[AggregatedCaseFields.CATEGORIES].apply(
        ast.literal_eval
    )
    df[AggregatedCaseFields.PRINCIPAL_LAWS] = df[
        AggregatedCaseFields.PRINCIPAL_LAWS
    ].apply(ast.literal_eval)
    df[AggregatedCaseFields.JURISDICTIONS] = df[
        AggregatedCaseFields.JURISDICTIONS
    ].apply(ast.literal_eval)
    return df
