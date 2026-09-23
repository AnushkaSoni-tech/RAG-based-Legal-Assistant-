# ⚖️ AI-Powered Legal Assistant for Indian Consumer Complaints

An AI-powered legal assistant that helps users understand and navigate consumer-related legal queries in India using Retrieval-Augmented Generation (RAG).

The system retrieves relevant legal information from the Consumer Protection Act, 2019 and uses an LLM to generate context-aware responses grounded in the retrieved legal content.

## 🚀 Live Demo

https://legal-advisorai.streamlit.app/

## 📌 Features

- 🔍 Legal document retrieval from the Consumer Protection Act, 2019
- 🤖 RAG-based question answering
- 🔎 Hybrid retrieval using dense and sparse search
- 📚 Context-grounded legal responses
- 💬 Interactive Streamlit interface
- 🇮🇳 India-specific consumer law focus
- 🔐 API keys handled through Streamlit Secrets

## 🧠 How It Works

```text
User Question
      ↓
Query Processing
      ↓
Hybrid Retrieval
(Dense + Sparse Search)
      ↓
Relevant Legal Context
      ↓
LLM Generation
      ↓
Final Answer
```

## 🛠️ Tech Stack

- Python
- Google Gemini
- Sentence Transformers
- Scikit-learn
- LangChain
- LangChain Community
- LangChain HuggingFace
- LangChain Text Splitters
- PyPDF
- Streamlit
- NumPy
- Pandas
- GitHub
- Streamlit Community Cloud

## 📂 Project Structure

```text
RAG-based-Legal-Assistant-/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── consumer_act.pdf
│
└── src/
    ├── __init__.py
    ├── pipeline.py
    ├── config.py
    ├── pdfExtraction.py
    │
    ├── chunking/
    │   └── chunking.py
    │
    └── retrieval/
        ├── dense_vec.py
        ├── sparse_vec.py
        └── hybrid_search.py
```

## 🔍 Retrieval Pipeline

### 1. Document Extraction
The Consumer Protection Act document is extracted from PDF format.

### 2. Text Chunking
The extracted legal text is divided into smaller chunks for retrieval.

### 3. Dense Retrieval
Sentence embeddings are used to capture semantic similarity between the query and legal text.

### 4. Sparse Retrieval
Keyword-based similarity helps identify relevant legal content.

### 5. Hybrid Retrieval
Dense and sparse retrieval results are combined to identify relevant legal context.

### 6. LLM Response Generation
The retrieved context is passed to the language model to generate the final response.

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/AnushkaSoni-tech/RAG-based-Legal-Assistant-.git
```

### 2. Navigate to the project

```bash
cd RAG-based-Legal-Assistant-
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 API Configuration

For local development, create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

Recommended `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

### Streamlit Cloud

Add the API key under:

**Streamlit Cloud → App Settings → Secrets**

Example:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Then access it in Python:

```python
import streamlit as st

api_key = st.secrets["GEMINI_API_KEY"]
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 💡 Example Questions

```text
What are the rights of a consumer under the Consumer Protection Act, 2019?
```

```text
What can a consumer do if a product purchased online is defective?
```

```text
Where can a consumer file a complaint?
```

```text
What is the procedure for filing a consumer complaint?
```

```text
What is considered a deficiency in service?
```

## ⚠️ Disclaimer

This application is an AI-based legal information tool intended for educational and informational purposes only.

It does not provide professional legal advice and should not be treated as a substitute for consultation with a qualified legal professional.

Users should verify important legal information with the relevant legislation or a qualified legal professional before taking legal action.

## 🔮 Future Improvements

- Improve retrieval accuracy
- Add more Indian legal acts and regulations
- Add conversation memory
- Improve legal source citations
- Add multilingual support
- Improve complex legal query handling
- Add retrieval and response evaluation
- Add user feedback mechanisms

## 👩‍💻 Author

**Anushka Soni**

B.Tech – Computer Science & Engineering (AI & ML)

**Technologies:** Python · Machine Learning · NLP · RAG · LangChain · LLMs · Streamlit
