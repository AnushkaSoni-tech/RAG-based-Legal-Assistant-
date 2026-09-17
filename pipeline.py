#importing modules
from retrieval.hybrid_search import hybrid_retrival
from chunking.chunking import chunking
from config import embedding_model
from google import genai
from config import prompt


#loading document
from pdfExtraction import pdf_extraction
documents = pdf_extraction("consumer_act.pdf")

#chunking
chunks=chunking(documents)

#loading key
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")


client=genai.Client(api_key=api_key)

def ask_question(question, chat_history=None):

    if chat_history is None:
        chat_history = []

    retrieved_documents = hybrid_retrival(
        embedding_model,
        chunks,
        question,
        k=4
    )

    retrieved_context = "\n\n".join(
        [
            f"Source {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    history = "\n".join(
        [
            f"{message['role']}:{message['content']}"
            for message in chat_history
        ]
    )

    formatted_prompt = prompt.format(
        history=history,
        retrieved_context=retrieved_context,
        question=question
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=formatted_prompt
    )

    answer = response.text

    return answer, retrieved_documents
