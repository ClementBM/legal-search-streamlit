from pathlib import Path

CLIMATE_CASES_CSV = (
    Path(__file__).parent.absolute() / "Global-Cases-Export-2023-08-18.csv"
)

ASCII_CLIMATE_CASES_CSV = Path(__file__).parent.absolute() / "ascii-global-cases.csv"
NORMED_CLIMATE_CASES_CSV = Path(__file__).parent.absolute() / "global-cases.csv"

VECTOR_INDEX_FOLDER_PATH = Path(__file__).parent.absolute() / "chroma_index"

WHOOSH_INDEX_FOLDER_PATH = Path(__file__).parent.absolute() / "whoosh_index"
