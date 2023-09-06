from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


from artefacts import INDEX_FOLDER_PATH


def load():
    # Retrieve embedding function from code env resources
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=repo_id)

    docsearch = Chroma(
        collection_name="climatecase",
        embedding_function=embeddings,
        persist_directory=str(INDEX_FOLDER_PATH),
    )
    return docsearch

    # docsearch.search("australia", search_type="similarity")
