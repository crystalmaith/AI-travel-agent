import os
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub  # Using Hugging Face LLMs
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# Other imports...
import os
import transformers  # Ensure transformers is updated

# Create vector store function
def create_vectorstore(travel_data):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Updated
    return embeddings

# Ensure tf-keras compatibility
try:
    import tf_keras as keras
except ImportError:
    print("⚠️ tf-keras not found! Run: pip install tf-keras")

# ✅ Load environment variables
dotenv_path = find_dotenv()
if not dotenv_path:
    raise ValueError("❌ No .env file found! Ensure it exists in the correct directory.")
load_dotenv(dotenv_path)

# ✅ Debug: Check if API key is loaded (for Hugging Face)
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise ValueError("❌ Hugging Face API Token not found. Add it to your .env file.")
else:
    print("✅ Hugging Face API Token Loaded.")

# ✅ Sample Travel Data (Placeholder)
travel_data = """
Paris: A beautiful city known for its art, culture, and the Eiffel Tower.
Tokyo: A vibrant city blending traditional temples with futuristic skyscrapers.
New York: A bustling metropolis famous for Times Square and Broadway.
"""

# ✅ Function to Process Travel Data
def create_vectorstore(data):
    """Splits text data and creates a FAISS vector store with HuggingFace embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    documents = text_splitter.create_documents([data])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# ✅ Initialize Vector Store
vectorstore = create_vectorstore(travel_data)

# ✅ Create RAG Pipeline using Hugging Face Model
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5}),
    retriever=vectorstore.as_retriever()
)

def travel_consultant(query):
    """Retrieve travel recommendations using the RAG model."""
    try:
        response = qa_chain.run(query)
        return response
    except Exception as e:
        return f"Error generating travel recommendation: {str(e)}"

# ✅ Example Usage (Test Cases)
if __name__ == "__main__":
    print("🗺 Travel Consultant:", travel_consultant("Recommend a city for a cultural experience"))
