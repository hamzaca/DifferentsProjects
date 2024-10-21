from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

import warnings

warnings.filterwarnings("ignore")


def embedding_function():
    return OllamaEmbeddings(model="llama3")


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


path = "LLMs/AskPDF/pdfs/"
file_path = "STaRSelf-TaughtReasoner.pdf"  # "AttentionsAllYouNeed.pdf"
# loader = PyPDFLoader(file_path=f"{path}/{file_path}")


doc_loader = PyPDFDirectoryLoader(path)
loaded_docucments = doc_loader.load()
chunks = split_documents(loaded_docucments)
print(chunks)

# # create Chroma Database
path_chroma_db = "chroma"
db = Chroma(persist_directory=path_chroma_db, embedding_function=embedding_function())
