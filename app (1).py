import streamlit as st

# IMPORTING LIBRARIES
from config import chunk_size, chunk_overlap, embedding_model, prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import google.generativeai as genai

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

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

# CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SIDEBAR
with st.sidebar:

    st.header("⚖️ Consumer Legal AI")

    st.write(
        "Ask questions related to consumer rights "
        "under the Consumer Protection Act, 2019."
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------
# RAG-PIPELINE
# --------------------------------------------------

# LOADING DOCUMENT
loader = PyPDFLoader("consumer_act.pdf")
documents = loader.load()


# CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(documents)


# EMBEDDING
model = embedding_model


# VECTOR STORE
vectorstore = InMemoryVectorStore(
    embedding=model
)

vectorstore.add_documents(
    documents=chunks
)


# --------------------------------------------------
# SPARSE VECTOR
# --------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vec = TfidfVectorizer()

tfidf_matrix = tfidf_vec.fit_transform(
    [chunk.page_content for chunk in chunks]
)


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

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


# --------------------------------------------------
# API KEY LOADING
# --------------------------------------------------

genai.configure(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# --------------------------------------------------
# DISPLAY PREVIOUS CHAT
# --------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

question = st.chat_input(
    "Describe your consumer issue..."
)


if question:

    # DISPLAY USER QUESTION
    with st.chat_message("user"):
        st.write(question)

    # RETRIEVAL
    retrieved_documents = hybrid_retrival(
        question,
        k=4
    )

    # CONVERT DOCUMENTS TO TEXT
    retrieved_context = "\n\n".join(
        [
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    # CONVERSATION HISTORY
    history = "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in st.session_state.chat_history
        ]
    )

    # PROMPT
    prompt = prompt.format(
        history=history,
        retrieved_context=retrieved_context,
        question=question
    )

    # GENERATION
    response = llm.generate_content(
        prompt
    )

    answer = response.text

    # DISPLAY AI RESPONSE
    with st.chat_message("assistant"):

        st.write(answer)

        # SHOW SOURCES
        with st.expander("📚 View Retrieved Legal Context"):

            for i, doc in enumerate(retrieved_documents):

                st.markdown(
                    f"### Source {i + 1}"
                )

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

    # SAVE USER MESSAGE
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # SAVE AI MESSAGE
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
