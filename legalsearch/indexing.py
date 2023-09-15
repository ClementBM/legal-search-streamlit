from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
import requests

from artefacts import (
    CLIMATE_CASES_CSV,
    VECTOR_INDEX_FOLDER_PATH,
    WHOOSH_INDEX_FOLDER_PATH,
    ASCII_CLIMATE_CASES_CSV,
    NORMED_CLIMATE_CASES_CSV,
)
from whoosh import scoring
from whoosh.fields import DATETIME, ID, STORED, TEXT, Schema, KEYWORD
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, syntax
from whoosh.qparser.plugins import MultifieldPlugin


from pathlib import Path
import pandas as pd
import unicodedata

MINILM_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_BERT_REPO_ID = "nlpaueb/legal-bert-base-uncased"


def download_csv(url, file_name):
    r = requests.get(url, allow_redirects=True)
    open(file_name, "wb").write(r.content)


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
    df = pd.read_csv(csv_path)

    df["Jurisdictions"] = df["Jurisdictions"].apply(preprocess_fields)
    df["Case Categories"] = df["Case Categories"].apply(preprocess_fields)
    df["Principal Laws"] = df["Principal Laws"].apply(preprocess_fields)
    df["Filing Year"] = pd.to_datetime(df["Filing Year"], format="%Y")

    df["Summary"] = df["Summary"].fillna("Not provided")

    df["Status"] = df["Status"].fillna("").astype(str)
    df["Status"] = df["Status"].astype("category")

    df["Filing Year"] = df["Filing Year"].fillna("")

    df = df.astype(str).replace("nan", "")
    return df[
        [
            "Title",
            "Summary",
            "Status",
            "Jurisdictions",
            "Case Categories",
            "Principal Laws",
            "Core Object",
            "Case Permalink",
        ]
    ]


def normalize_file(raw_file, clean_file):
    with open(raw_file, "r+", encoding="utf-8") as file_reader:
        content = file_reader.read()
        ascii_content = unicodedata.normalize("NFKD", content).encode("ascii", "ignore")

    with open(clean_file, "wb") as file_writer:
        file_writer.write(ascii_content)


def prepare_csv_file():
    global_climate_change_litigation_csv = "https://climatecasechart.com/wp-content/uploads/2023/08/Global-Cases-Export-2023-08-18.csv"

    if not Path(CLIMATE_CASES_CSV).exists():
        download_csv(global_climate_change_litigation_csv, CLIMATE_CASES_CSV)

    if not Path(ASCII_CLIMATE_CASES_CSV).exists():
        normalize_file(CLIMATE_CASES_CSV, ASCII_CLIMATE_CASES_CSV)

    if not Path(NORMED_CLIMATE_CASES_CSV).exists():
        load_global_cases(ASCII_CLIMATE_CASES_CSV).to_csv(
            NORMED_CLIMATE_CASES_CSV, index=False
        )


def build_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name=LEGAL_BERT_REPO_ID)

    loader = CSVLoader(str(NORMED_CLIMATE_CASES_CSV), source_column="Case Permalink")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    docsearch = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="climatecase",
        persist_directory=str(VECTOR_INDEX_FOLDER_PATH),
    )
    return docsearch


def load_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name=LEGAL_BERT_REPO_ID)

    docsearch = Chroma(
        collection_name="climatecase",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_INDEX_FOLDER_PATH),
    )
    return docsearch


bm25_index = open_dir(WHOOSH_INDEX_FOLDER_PATH)
climate_cases = pd.read_csv(CLIMATE_CASES_CSV)
vector_index = vector_index = load_vector_index()

SCHEMA = Schema(
    title=TEXT(stored=True),
    summary=TEXT(stored=True),
    permalink=STORED(),
    jurisdictions=TEXT(stored=True),
    principal_laws=TEXT(stored=True),
    case_categories=TEXT(stored=True),
    status=TEXT(stored=True),
    filing_year=DATETIME(stored=True),
)


def MyParser(fieldnames, schema, fieldboosts=None):
    p = QueryParser(None, schema, group=syntax.OrGroup)
    mfp = MultifieldPlugin(fieldnames, fieldboosts=fieldboosts)
    p.add_plugin(mfp)
    return p


def bm25_search(query, top_k=10):
    query_parser = MyParser(
        fieldnames=["title", "summary"],
        schema=SCHEMA,
    ).parse(query)

    with bm25_index.searcher(weighting=scoring.BM25F()) as searcher:
        results = searcher.search(
            q=query_parser,
            limit=top_k,
            terms=True,
        )

    doc_ids = [id for score, id in results.top_n]
    return doc_ids
