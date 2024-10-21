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


def calculate_chunk_ids(chunks):

    # This will create IDs like "pdfs/AttentionsAllYouNeed.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


folder_path = "LLMs/AskPDF/pdfs/"
file_path = "STaRSelf-TaughtReasoner.pdf"  # "AttentionsAllYouNeed.pdf"
# loader = PyPDFLoader(file_path=f"{path}/{file_path}")


doc_loader = PyPDFDirectoryLoader(folder_path)
loaded_docucments = doc_loader.load()
chunks = split_documents(loaded_docucments)

print("***" * 50)
print("***" * 50)
print(chunks[0])
print("***" * 50)
print(chunks[0].metadata)


# ------------------------------------------------ Add to Chroma --------------------------------------------------

# https://docs.trychroma.com/getting-started

# # create Chroma Database
# path_chroma_db = "chroma"
# db = Chroma(persist_directory=path_chroma_db, embedding_function=embedding_function())

chunks_ids = calculate_chunk_ids(chunks)


# Add or Update the documents.
existing_items = db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in DB: {len(existing_ids)}")


# Only add documents that don't exist in the DB.
new_chunks = []
for chunk in chunks_with_ids:
    if chunk.metadata["id"] not in existing_ids:
        new_chunks.append(chunk)

if len(new_chunks):
    print(f"👉 Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
else:
    print("✅ No new documents to add")
# --------------------------------------------------------------------------------------------------
