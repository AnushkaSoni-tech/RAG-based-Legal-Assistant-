from langchain_core.vectorstores import InMemoryVectorStore

def dense_vec(embedding_model , chunks):
    vectorstore=InMemoryVectorStore(embedding=embedding_model)
    vectorstore.add_documents(documents=chunks)
    return vectorstore

