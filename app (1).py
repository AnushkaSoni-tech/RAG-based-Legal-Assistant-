import streamlit as st

# IMPORTING LIBRARIES
from config import chunk_size, chunk_overlap, embedding_model, prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG
st.set_page_config(
    page_title="AI-Powered legal Assistant for Indian Consumer Complaints",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown(
    """
<style>

/* MAIN APP */
.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef2f7 100%
    );
}


/* MAIN CONTENT */
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
    font-size: 1.7rem;
    font-weight: 750;
    margin-bottom: 0.8rem;
    color: white;
}

.sidebar-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.7;
}

.hero-container {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a5f
    );

    padding: 2.5rem;

    border-radius: 22px;

    margin-bottom: 2rem;

    box-shadow:
        0 10px 30px rgba(
            15,
            23,
            42,
            0.18
        );
}


.hero-badge {
    display: inline-block;

    background:
        rgba(
            255,
            255,
            255,
            0.12
        );

    color: #e2e8f0;

    padding: 8px 16px;

    border-radius: 30px;

    font-size: 0.85rem;

    font-weight: 600;

    margin-bottom: 18px;

    letter-spacing: 0.3px;
}


.hero-title {
    font-size: 3rem;

    font-weight: 800;

    color: white;

    letter-spacing: -1px;

    margin-bottom: 12px;
}


.hero-subtitle {
    font-size: 1.08rem;

    color: #cbd5e1;

    line-height: 1.7;

    max-width: 720px;
}


.hero-subtitle b {
    color: white;
}


/* SECTION HEADING */
.section-heading {
    font-size: 1.35rem;

    font-weight: 700;

    color: #0f172a;

    margin-top: 1.5rem;

    margin-bottom: 1rem;
}


/*INFORMATION CARDS */

.info-card {
    background: white;

    padding: 1.5rem;

    border-radius: 18px;

    border:
        1px solid #e2e8f0;

    box-shadow:
        0 6px 18px rgba(
            0,
            0,
            0,
            0.06
        );

    height: 100%;

    min-height: 150px;

    transition: 0.2s;
}


.info-card:hover {
    transform:
        translateY(-3px);

    box-shadow:
        0 10px 24px rgba(
            0,
            0,
            0,
            0.10
        );
}


.card-title {
    font-size: 1.1rem;

    font-weight: 700;

    color: #0f172a;

    margin-bottom: 0.8rem;
}


.card-text {
    font-size: 0.92rem;

    color: #64748b;

    line-height: 1.6;
}


/* WELCOME BOX*/
.welcome-box {
    background: white;

    border-radius: 20px;

    padding: 2.5rem;

    border:
        1px solid #e2e8f0;

    text-align: center;

    box-shadow:
        0 6px 20px rgba(
            0,
            0,
            0,
            0.05
        );

    margin-top: 2rem;
}


.welcome-icon {
    font-size: 3rem;

    margin-bottom: 0.7rem;
}


.welcome-title {
    font-size: 1.45rem;

    font-weight: 700;

    color: #0f172a;

    margin-bottom: 0.7rem;
}


.welcome-text {
    color: #64748b;

    line-height: 1.7;
}


/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background-color:
        #ffffff !important;

    color:
        #111827 !important;

    border-radius:
        18px;

    padding:
        1.2rem;

    margin-bottom:
        1rem;

    border:
        1px solid #dbe3ee;

    box-shadow:
        0 4px 15px rgba(
            15,
            23,
            42,
            0.06
        );
}


/* TEXT INSIDE CHAT */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] .stMarkdown {
    color:
        #111827 !important;
}


/* HEADINGS */

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color:
        #0f172a !important;
}


/* BOLD TEXT */

[data-testid="stChatMessage"] strong {
    color:
        #0f172a !important;
}


/* LINKS */

[data-testid="stChatMessage"] a {
    color:
        #2563eb !important;
}


/*  CHAT INPUT */

[data-testid="stChatInput"] {
    background:
        transparent !important;
}


[data-testid="stChatInput"] textarea {
    background-color:
        #ffffff !important;

    color:
        #111827 !important;

    -webkit-text-fill-color:
        #111827 !important;

    caret-color:
        #111827 !important;

    border:
        1px solid #cbd5e1 !important;

    border-radius:
        14px !important;

    font-size:
        16px !important;
}


[data-testid="stChatInput"] textarea::placeholder {
    color:
        #94a3b8 !important;

    opacity:
        1 !important;
}


[data-testid="stChatInput"] > div {
    color:
        #111827 !important;
}


[data-testid="stChatInput"] button {
    border-radius:
        12px !important;
}

.stButton > button {
    width: 100%;

    border-radius: 10px;

    border: none;

    padding: 0.65rem;

    font-weight: 600;

    transition: 0.2s;
}


.stButton > button:hover {
    transform:
        translateY(-1px);
}


[data-testid="stExpander"] {
    background-color:
        #ffffff !important;

    border:
        1px solid #dbe3ee !important;

    border-radius:
        12px !important;

    margin-top:
        10px !important;

    overflow:
        hidden !important;
}


[data-testid="stExpander"] summary {
    color:
        #0f172a !important;

    font-weight:
        600 !important;

    background-color:
        #f8fafc !important;
}


[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] li {
    color:
        #111827 !important;
}


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

# HEADER
st.markdown(
    """
<div class="hero-container">

<div class="hero-badge">
⚖️ AI-Powered Legal Assistance
</div>

<div class="hero-title">
Consumer Legal AI
</div>

<div class="hero-subtitle">
Get AI-powered assistance for consumer rights and legal concerns based on
<b>The Consumer Protection Act, 2019</b>.
</div>

</div>
    """,
    unsafe_allow_html=True
)

# CHAT HISTORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SIDEBAR
with st.sidebar:

    st.markdown(
        """
<div class="sidebar-title">
⚖️ Consumer Legal AI
</div>

<div class="sidebar-subtitle">
Your AI assistant for understanding consumer rights under the
Consumer Protection Act, 2019.
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
    
# QUICK INFORMATION CARDS
if len(st.session_state.chat_history) == 0:

    st.markdown(
        """
<div class="section-heading">
How can I help you today?
</div>
        """,
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
Understand your rights when you receive defective,
damaged, or incorrect products.
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
Learn about consumer rights related to refunds,
replacements, and cancellations.
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
Understand possible remedies and actions available
under consumer protection law.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


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



# RAG PIPELINE

# LOADING DOCUMENT
loader = PyPDFLoader(
    "consumer_act.pdf"
)

documents = loader.load()


# CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(
    documents
)


# EMBEDDING
model = embedding_model


# VECTOR STORE
vectorstore = InMemoryVectorStore(
    embedding=model
)

vectorstore.add_documents(
    documents=chunks
)

# SPARSE VECTOR
tfidf_vec = TfidfVectorizer()

tfidf_matrix = tfidf_vec.fit_transform(
    [
        chunk.page_content
        for chunk in chunks
    ]
)

# HYBRID RETRIEVAL
def hybrid_retrival(query, k=4):

    # DENSE RETRIEVAL
    dense_doc = vectorstore.similarity_search(
        query,
        k=k
    )


    # TF-IDF RETRIEVAL
    query_ = tfidf_vec.transform(
        [query]
    )

    score = cosine_similarity(
        query_,
        tfidf_matrix
    )[0]


    # TOP K TF-IDF CHUNKS
    top_indices = score.argsort()[-k:][::-1]

    tfidf_doc = [
        chunks[i]
        for i in top_indices
    ]


    # COMBINE RESULTS
    combine_doc = []

    for doc in dense_doc + tfidf_doc:

        if doc not in combine_doc:

            combine_doc.append(
                doc
            )


    return combine_doc[:k]

# API KEY LOADING
genai.configure(
    api_key=st.secrets["GOOGLE_API_KEY"]
)


llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# CHAT AREA TITLE
st.markdown(
    """
<div class="section-heading">
💬 Legal Assistant
</div>
    """,
    unsafe_allow_html=True
)

# DISPLAY PREVIOUS CHAT
for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


        # DISPLAY SAVED SOURCES

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.markdown(
                "### 📚 Retrieved Legal Sources"
            )


            for i, source_data in enumerate(
                message["sources"]
            ):

                with st.expander(
                    f"📄 Source {i + 1} — "
                    "Click to view legal context",
                    expanded=False
                ):

                    st.markdown(
                        f"**Page:** "
                        f"{source_data['page']}"
                    )

                    st.markdown(
                        f"**Document:** "
                        f"{source_data['document']}"
                    )

                    st.markdown(
                        "#### Legal Text"
                    )

                    st.write(
                        source_data["content"]
                    )

# USER QUERY
question = st.chat_input(
    "Describe your consumer issue..."
)


if question:

    # DISPLAY USER QUESTION
    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )
    # RETRIEVAL
    retrieved_documents = hybrid_retrival(
        question,
        k=4
    )


    # CONVERT DOCUMENTS TO TEXT
    retrieved_context = "\n\n".join(
        [
            f"[Source {i + 1}]\n"
            f"{doc.page_content}"

            for i, doc in enumerate(
                retrieved_documents
            )
        ]
    )

    # CONVERSATION HISTORY
    history = "\n".join(
        [
            f"{message['role']}: "
            f"{message['content']}"

            for message in
            st.session_state.chat_history
        ]
    )
    # PROMPT
    final_prompt = prompt.format(
        history=history,
        retrieved_context=retrieved_context,
        question=question
    )
    # GENERATION
    response = llm.generate_content(
        final_prompt
    )

    answer = response.text
    # DISPLAY AI RESPONSE
    with st.chat_message(
        "assistant"
    ):

        st.write(
            answer
        )


        st.markdown(
            "### 📚 Retrieved Legal Sources"
        )


        for i, doc in enumerate(
            retrieved_documents
        ):

            page = doc.metadata.get(
                "page_label",
                doc.metadata.get(
                    "page",
                    "Not available"
                )
            )


            source = doc.metadata.get(
                "source",
                "Consumer Protection Act, 2019"
            )


            with st.expander(
                f"📄 Source {i + 1} — "
                "Click to view legal context",
                expanded=False
            ):

                st.markdown(
                    f"**Page:** {page}"
                )

                st.markdown(
                    f"**Document:** {source}"
                )

                st.markdown(
                    "#### Legal Text"
                )

                st.write(
                    doc.page_content
                )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",

            "content": answer,

            "sources": [

                {
                    "page": doc.metadata.get(
                        "page_label",
                        doc.metadata.get(
                            "page",
                            "Not available"
                        )
                    ),

                    "document": doc.metadata.get(
                        "source",
                        "Consumer Protection Act, 2019"
                    ),

                    "content": doc.page_content
                }

                for doc in retrieved_documents
            ]
        }
    )
