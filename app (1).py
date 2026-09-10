import streamlit as st

# IMPORTING LIBRARIES
from config import chunk_size, chunk_overlap, embedding_model, prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import google.generativeai as genai


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Consumer Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2f7 100%
        );
    }


    /* REMOVE DEFAULT TOP SPACE */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }


    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #1e293b 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }


    /* SIDEBAR TITLE */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sidebar-subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.6;
    }


    /* HERO SECTION */
    .hero-container {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a5f
        );
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.6;
    }


    /* INFO CARDS */
    .info-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
    }

    .card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }

    .card-text {
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.5;
    }


    /* CHAT MESSAGE */
    [data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }


    /* CHAT INPUT */
    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: white !important;
    }


    /* BUTTON */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.6rem;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }


    /* EXPANDER */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 10px;
        font-weight: 600;
    }


    /* SECTION HEADINGS */
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }


    /* WELCOME MESSAGE */
    .welcome-box {
        background: white;
        border-radius: 18px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.04);
        margin-top: 1rem;
    }

    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .welcome-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }

    .welcome-text {
        color: #64748b;
        margin-top: 0.5rem;
        line-height: 1.6;
    }


    /* HIDE STREAMLIT BRANDING */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER / HERO SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-container">

        <div class="hero-title">
            ⚖️ Consumer Legal AI
        </div>

        <div class="hero-subtitle">
            Get AI-powered assistance for consumer rights and legal concerns
            based on the Consumer Protection Act, 2019.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ⚖️ Consumer Legal AI
        </div>

        <div class="sidebar-subtitle">
            Your AI assistant for understanding consumer rights
            under the Consumer Protection Act, 2019.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 💡 What you can ask")

    st.caption("• Defective products")
    st.caption("• Refund and replacement issues")
    st.caption("• Online shopping problems")
    st.caption("• Misleading advertisements")
    st.caption("• Consumer complaints")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption(
        "⚠️ This AI provides legal information for educational purposes "
        "and should not be considered professional legal advice."
    )


# --------------------------------------------------
# QUICK INFORMATION CARDS
# --------------------------------------------------

if len(st.session_state.chat_history) == 0:

    st.markdown(
        '<div class="section-heading">How can I help you today?</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">
                    🛍️ Product Issues
                </div>

                <div class="card-text">
                    Understand your rights when you receive
                    defective, damaged, or incorrect products.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">
                    💳 Refund Problems
                </div>

                <div class="card-text">
                    Learn about consumer rights related to
                    refunds, replacements, and cancellations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">
                    📢 Consumer Complaints
                </div>

                <div class="card-text">
                    Understand possible remedies and actions
                    available under consumer protection law.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="welcome-box">

            <div class="welcome-icon">
                ⚖️
            </div>

            <div class="welcome-title">
                Describe your consumer issue
            </div>

            <div class="welcome-text">
                Tell me what happened, and I will analyze your query
                using the Consumer Protection Act, 2019.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


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
# CHAT AREA TITLE
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">💬 Legal Assistant</div>',
    unsafe_allow_html=True
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
