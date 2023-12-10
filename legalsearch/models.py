from enum import Enum


class GlobalCaseFields:
    """
    Column names of the raw Global-Case csv file.
    """

    ID: str = "ID"
    TITLE: str = "Case Name"
    SUMMARY: str = "Summary"
    STATUS: str = "Status"
    JURISDICTIONS: str = "Jurisdictions"
    CATEGORIES: str = "Case Categories"
    PRINCIPAL_LAWS: str = "Principal Laws"
    FILING_YEAR: str = "Filing Year for Action"
    PERMALINK: str = "Permalink"
    REPORTER_INFO: str = "Reporter Info"
    CORE_OBJECT: str = "Core Object"


class UsCaseFields:
    """
    Column names of the raw US-Case csv file.
    """

    ID: str = "ID"
    TITLE: str = "Case Name"
    SUMMARY: str = "Description"
    FILING_YEAR: str = "Filing Year"
    CATEGORIES: str = "Case Categories"
    PRINCIPAL_LAWS: str = "Principal Laws"


class AggregatedCaseFields:
    """ """

    TITLE: str = "Title"
    SUMMARY: str = "Summary"
    STATUS: str = "Status"
    JURISDICTIONS: str = "Jurisdictions"
    CATEGORIES: str = "Case Categories"
    PRINCIPAL_LAWS: str = "Principal Laws"
    FILING_YEAR: str = "Filing Year"
    PERMALINK: str = "Permalink"

    SEARCHABLE_CONTENT: str = "columns_concatenations"

    COLUMNS: list = [
        TITLE,
        SUMMARY,
        STATUS,
        JURISDICTIONS,
        CATEGORIES,
        PRINCIPAL_LAWS,
        FILING_YEAR,
        PERMALINK,
        SEARCHABLE_CONTENT,
    ]
