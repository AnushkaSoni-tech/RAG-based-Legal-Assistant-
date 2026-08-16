import streamlit as st

#IMPORTING LIBRARIES
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# STREAMLIT UI
st.set_page_config(
    page_title="Consumer Legal AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Consumer Legal AI")

st.write(
    "AI-powered legal assistance based on the "
    "Consumer Protection Act, 2019."
)

st.info(
    "⚠️ This AI assistant provides general legal information "
    "and is not a substitute for professional legal advice."
)


# RAG-PIPELINE

# LOADING DOCUMENT
loader = PyPDFLoader("consumer_act.pdf")
documents = loader.load()

# CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)

# EMBEDDING
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# VECTOR STORE
vectorstore = InMemoryVectorStore(
    embedding=model
)

vectorstore.add_documents(
    documents=chunks
)


# RETRIEVER
retriever = vectorstore.as_retriever()


# API KEY LOADING
genai.configure(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# USER QUERY
question = st.chat_input(
    "Describe your consumer issue..."
)


if question:

    # RETRIEVAL
    retrieved_documents = retriever.invoke(question)

    #PROMPT
    response = llm.generate_content(
        f"""You are an AI legal assistant for the Consumer Protection Act, 2019.

Instructions:
1. Answer ONLY using the retrieved context.
2. First identify the consumer's issue.
3. Explain how the retrieved legal provisions apply to the user's situation.
4. Mention the relevant sections used.
5. If the context supports it, explain the remedies available.
6. Do not include unrelated provisions.
7. If the context is insufficient, explicitly state that additional legal provisions are needed.
8. Never invent laws or sections.

retrieved context:{retrieved_documents}

questions:{question}

Your citations MUST include:
- Document name
- Page number
- Section/article if available
- Source/file name

Citation format:

[Source: <document name>, Page: <page_label>, Section: <section>]

If a particular metadata field is unavailable, write "Not available".

DO NOT invent page numbers, sections, document names, or other metadata.

If multiple pieces of information come from different pages, cite each relevant page separately.
"""
    )


   
    # DISPLAY ANSWER
    st.markdown("### ⚖️ Legal Assistant")

    st.write(response.text)

    # SHOW RETRIEVED DOCUMENTS
    with st.expander("📚 View Retrieved Legal Context"):

        for i, doc in enumerate(retrieved_documents):

            st.markdown(f"### Source {i + 1}")

            st.write(
                f"**Page:** "
                f"{doc.metadata.get('page_label', 'Not available')}"
            )

            st.write(
                f"**Document:** "
                f"{doc.metadata.get('source', 'Not available')}"
            )

            st.write(doc.page_content)

            st.divider()
