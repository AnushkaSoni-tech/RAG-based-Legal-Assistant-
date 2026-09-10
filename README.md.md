# AI-Powered Legal Assistant

An AI-powered legal assistant designed to help users understand and find relevant provisions from the **Consumer Protection Act, 2019**.

The project uses **Retrieval-Augmented Generation (RAG)** with a **hybrid retrieval approach**, combining dense vector search and TF-IDF to retrieve relevant legal information before generating an answer.

## 🚀 Live Demo

**Try the application:**  
https://consumer-legal-assistant.streamlit.app/

## ✨ Features

- **RAG-based legal question answering** using the Consumer Protection Act, 2019
- **Hybrid retrieval** using dense vector search + TF-IDF
- **Context-grounded responses** generated with Gemini 2.5 Flash
- **Source-aware answers** based on retrieved legal content
- **PDF processing** for the legal knowledge base
- **Vector embeddings and vector storage**
- **Streamlit interface** for an interactive user experience
- Displays the **retrieved context** alongside the generated answer

## 🧠 How It Works

```text
User Question
      ↓
Query Processing
      ↓
Hybrid Retrieval
 ┌───────────────┐
 │ Vector Search │
 │    +          │
 │    TF-IDF     │
 └───────────────┘
      ↓
Relevant Legal Context
      ↓
Context-Grounded Prompt
      ↓
Gemini 2.5 Flash
      ↓
Answer + Retrieved Context
```

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **Gemini 2.5 Flash**
- **Retrieval-Augmented Generation (RAG)**
- **Vector Embeddings**
- **TF-IDF**
- **Streamlit**
- **PDF Processing**
- **Vector Storage**

## 📌 Project Highlights

- Built a legal question-answering system focused on the **Consumer Protection Act, 2019**.
- Implemented **hybrid retrieval** to improve the relevance of retrieved legal provisions.
- Used **Gemini 2.5 Flash** with context-grounded prompting to generate source-aware responses.
- Developed an interactive **Streamlit** application for processing documents, storing embeddings, retrieving context, and displaying answers.

## ⚠️ Disclaimer

This project is intended for **educational and informational purposes only**. It is not a substitute for professional legal advice.

## 🔗 Demo

Try it here:

https://consumer-legal-assistant.streamlit.app/
