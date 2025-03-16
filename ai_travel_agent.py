import openai
import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI

import os
from dotenv import load_dotenv, find_dotenv

# Find .env file and load it
dotenv_path = find_dotenv()
if not dotenv_path:
    raise ValueError("❌ No .env file found! Ensure it exists in the correct directory.")

load_dotenv(dotenv_path)

# Debug: Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API Key not found. Check .env file or environment variables.")
else:
    print(f"✅ API Key Loaded: {api_key[:5]}... (truncated for security)")

# ✅ Sample Travel Data (Placeholder)
travel_data = """
Paris: A beautiful city known for its art, culture, and the Eiffel Tower.
Tokyo: A vibrant city blending traditional temples with futuristic skyscrapers.
New York: A bustling metropolis famous for Times Square and Broadway.
"""

# ✅ Function to Process Travel Data
def create_vectorstore(data):
    """Splits text data and creates a FAISS vector store with OpenAI embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    documents = text_splitter.create_documents([data])
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# ✅ Initialize Vector Store
vectorstore = create_vectorstore(travel_data)

# ✅ Create RAG Pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), retriever=vectorstore.as_retriever()
)

def travel_consultant(query):
    """Retrieve travel recommendations using the RAG model."""
    try:
        response = qa_chain.run(query)
        return response
    except Exception as e:
        return f"Error generating travel recommendation: {str(e)}"

def sales_consultant(user_inquiry):
    """Generate responses for travel sales inquiries using OpenAI's GPT model."""
    prompt = f"Act as a travel sales consultant and answer this query: {user_inquiry}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel sales consultant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except openai.OpenAIError as e:
        return f"OpenAI API error: {str(e)}"

# ✅ Example Usage (Test Cases)
if __name__ == "__main__":
    print("🗺 Travel Consultant:", travel_consultant("Recommend a city for a cultural experience"))
    print("💰 Sales Consultant:", sales_consultant("What are the best deals for a Paris vacation?"))
