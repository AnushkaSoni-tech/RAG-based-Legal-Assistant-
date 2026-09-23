from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import chunk_size,chunk_overlap
def chunking(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks=text_splitter.split_documents(documents)
    return chunks

