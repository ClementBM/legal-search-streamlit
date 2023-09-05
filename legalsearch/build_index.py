from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma

from legalsearch.artefacts import CLIMATE_CASES_CSV, INDEX_FOLDER_PATH


def build():
    # Retrieve embedding function from code env resources
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=repo_id)

    loader = CSVLoader(str(CLIMATE_CASES_CSV), source_column="Case Permalink")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    docsearch = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="climatecase",
        persist_directory=str(INDEX_FOLDER_PATH),
    )
    return docsearch
