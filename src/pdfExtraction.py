from langchain_community.document_loaders import PyPDFLoader

def pdf_extraction(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents