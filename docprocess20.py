import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever 
from langchain.chains.query_constructor.base import AttributeInfo

# Define file path variable
FILE_PATH = 'E:/cidc/sample_text.txt' 

# Global variables for vector store, embeddings, and LLM
vector_store = None  
embeddings = None  
llm = None  # Initialize LLM variable

def load_env_variables():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv
    load_dotenv('config.env')

def initialize_llm():
    """Initialize the LLM using the Groq API key."""
    global llm  # Declare llm as global
    groq_api_key = os.getenv('groq_api_key')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma2-9b-it', temperature=0.1)

def clean_timestamps(content):
    """Clean timestamps from a text."""
    timestamp_pattern = r'\b\d{2}:\d{2}:\d{2}\.\d{3}\b'
    cleaned_content = re.sub(timestamp_pattern, '', content)
    return re.sub(r'\s+', ' ', cleaned_content).strip()

def split_docs(documents, chunk_size=1000, chunk_overlap=50):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def initialize_embeddings_and_vector_store(index_name):
    """Initialize embeddings and Pinecone vector store."""
    global vector_store, embeddings  

    if embeddings is None:
        google_api_key = os.getenv('GEMINI_API_KEY')
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/text-embedding-004")
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pc = pinecone.Pinecone(api_key=pinecone_api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, dimension=768)

        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    return embeddings  

def add_documents_to_vector_store(documents):
    """Add documents to the Pinecone vector store using the global vector_store."""
    global vector_store  
    vector_store.add_documents(documents)

def get_similar_docs(query, k=1):
    """Retrieve similar documents based on a query using the global vector_store."""
    global vector_store  
    return vector_store.similarity_search(query, k=k)

def read_file_content(file_path):
    """Read content from a specified file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def assign_topics_and_main_topics(documents):
    """Assign topics and main topics to each document chunk using LLM."""
    
    for doc in documents:
        formatted_input_topic = f"Topic should be of one or two words only. Identify a topic for the following text: {doc.page_content}"
        response_topic = llm.invoke(formatted_input_topic)
        topic = response_topic.content.strip()
        
        formatted_input_main_topic = (
            f"Given the topic '{topic}', identify a broader main topic. Don't give explanation for main topic. Just identify the main topic only."
        )
        response_main_topic = llm.invoke(formatted_input_main_topic)
        main_topic = response_main_topic.content.strip()
        
        doc.metadata['topic'] = topic
        doc.metadata['main_topic'] = main_topic

def retrieve_relevant_documents(user_query):
    """Retrieve relevant documents based on user query."""
    global vector_store  
    if vector_store:
        metadata_field_info = [
            AttributeInfo(name="main_topic", description="Main topic of the chunk", type="string or list[string]"),
            AttributeInfo(name="text", description="Content present inside the chunk", type="string"),
            AttributeInfo(name="topic", description="Topic of the chunk", type="string")
        ]

        document_content_description = "Brief conversational file"
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
        )
        
        return retriever.get_relevant_documents(user_query)
    
    return []

def process_documents(file_path, index_name):
    """Process documents: clean, split, embed, assign topics, and add to Pinecone."""
    load_env_variables()
    
    content = read_file_content(file_path)
    cleaned_content = clean_timestamps(content)  

    documents = [Document(page_content=cleaned_content)]
    
    documents = split_docs(documents)

    initialize_embeddings_and_vector_store(index_name)

    assign_topics_and_main_topics(documents)

    add_documents_to_vector_store(documents)

# Load environment variables and initialize LLM before processing documents
load_env_variables()
initialize_llm()

# Call this function with appropriate parameters when you want to process your documents.
process_documents(FILE_PATH, 'testproject4')
