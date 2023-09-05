from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings

from artefacts import INDEX_FOLDER_PATH
from langchain.vectorstores import Chroma


def load():
    # Retrieve embedding function from code env resources
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceHubEmbeddings(repo_id=repo_id)

    docsearch = Chroma(
        collection_name="climatecase",
        embedding_function=embeddings,
        persist_directory=str(INDEX_FOLDER_PATH),
    )
    return docsearch

    # docsearch.search("australia", search_type="similarity")
