import streamlit as st

# IMPORTING LIBRARIES
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


# SPARSE VECTOR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vec = TfidfVectorizer()

tfidf_matrix = tfidf_vec.fit_transform(
    [chunk.page_content for chunk in chunks]
)


# HYBRID RETRIEVAL
def hybrid_retrival(query, k=4):

    # Dense
    dense_doc = vectorstore.similarity_search(
        query,
        k=k
    )

    # TF-IDF
    query_ = tfidf_vec.transform([query])

    score = cosine_similarity(
        query_,
        tfidf_matrix
    )[0]

    # Top k TF-IDF chunks
    top_indices = score.argsort()[-k:][::-1]

    tfidf_doc = [
        chunks[i]
        for i in top_indices
    ]

    # Combine
    combine_doc = []

    for doc in dense_doc + tfidf_doc:

        if doc not in combine_doc:
            combine_doc.append(doc)

    return combine_doc[:k]


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
    retrieved_documents = hybrid_retrival(
        question,
        k=4
    )

    # PROMPT
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


Your answer must use simple source labels only.

For information supported by the first retrieved source, write:
[Source 1]

For information supported by the second retrieved source, write:
[Source 2]

For information supported by the third retrieved source, write:
[Source 3]

Continue numbering the sources in the same order as they appear
in the retrieved context.

DO NOT include:
- UUIDs
- document IDs
- file names
- page numbers
- metadata
- long source identifiers

Use ONLY:
[Source 1]
[Source 2]
[Source 3]
etc.

If one statement is supported by multiple sources, write:
[Source 1][Source 2]

Do not invent sources. Use only the sources provided in the retrieved context..
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
