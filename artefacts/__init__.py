from pathlib import Path

# Raw files
RAW_GLOBAL_CASES_CSV = (
    Path(__file__).parent.absolute() / "Global-Cases-Export-2023-11-20.csv"
)
RAW_US_CASES_CSV = Path(__file__).parent.absolute() / "US-Case-Bundles-2023-11-20.csv"

# Normalized Files
ASCII_ALL_CASES_CSV = Path(__file__).parent.absolute() / "all-cases-ascii.csv"
NORMED_ALL_CASES_CSV = Path(__file__).parent.absolute() / "all-cases-normed.csv"

# Index Files
VECTOR_INDEX_FOLDER_PATH = Path(__file__).parent.absolute() / "chroma_index"
WHOOSH_INDEX_FOLDER_PATH = Path(__file__).parent.absolute() / "whoosh_index"

# Aggregated data files
ALL_CASES_CSV = Path(__file__).parent.absolute() / "all-cases.csv"
