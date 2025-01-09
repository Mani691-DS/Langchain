## Inference
`docprocess20.py`, automates the processing of text files for semantic understanding and retrieval. It integrates document cleaning, chunking, embedding generation, topic and main topic assignment, and storing/retrieving content using Pinecone as a vector database. Below is an overview:

1. **Environment Setup**:
   - Loads API keys for Groq LLM, Google Generative AI, and Pinecone from a `.env` file.
   
2. **Functionality**:
   - **Text Preprocessing**: Cleans unnecessary timestamps and formats content.
   - **Document Chunking**: Splits large documents into smaller chunks using `RecursiveCharacterTextSplitter`.
   - **Embeddings and Vector Store**:
     - Initializes GoogleGenerativeAI embeddings.
     - Configures and uses Pinecone to store document vectors.
   - **Topic Assignment**:
     - Uses Groq LLM to generate topics and main topics for each chunk of text.
   - **Content Storage and Retrieval**:
     - Adds processed chunks to the Pinecone vector database.
     - Retrieves relevant documents based on user queries using `SelfQueryRetriever`.

3. **How to Use**:
   - Specify the file path (`FILE_PATH`) and the Pinecone index name (`index_name`).
   - Call `process_documents()` to preprocess, analyze, and store the text for semantic retrieval.

4. **Technologies Used**:
   - **LangChain** for document chunking and query retrieval.
   - **Pinecone** for vector storage and similarity search.
   - **Google Generative AI** for embeddings.
   - **Groq LLM** for topic and main topic extraction.

This script is ideal for applications requiring automated text classification, semantic search, and topic-driven organization of documents.


`test20nov.py`, creates an interactive question-answering chatbot application using Streamlit, LangChain, and Groq LLM. It integrates conversation memory and vector-based semantic search for precise and context-aware responses. Below is an overview:

1. **Environment Setup**:
   - Loads API keys for Groq LLM and initializes the LangChain-based Q&A bot.

2. **Functionality**:
   - **User Interaction**: 
     - Accepts user queries via a text input field.
     - Displays chat history in a user-friendly interface using `streamlit_chat`.
   - **Semantic Search**: 
     - Uses `retrieve_relevant_documents()` from `docprocess20.py` to fetch relevant information from a Pinecone vector database.
   - **LLM Response**:
     - Uses Groq LLM to refine and provide query-specific answers based on retrieved documents.
   - **Conversation Memory**:
     - Maintains conversation history using LangChain's `ConversationBufferWindowMemory` to provide context-aware responses.

3. **Streamlit Components**:
   - **Chat History**: Displays the conversation between the user and the bot.
   - **Text Input**: Allows the user to input queries for processing.

4. **Technologies Used**:
   - **Streamlit** for building the chatbot interface.
   - **LangChain** for memory management, prompting, and conversation handling.
   - **Groq LLM** for generating accurate and contextually relevant answers.
   - **Pinecone** (via `retrieve_relevant_documents`) for retrieving relevant content.

This script is ideal for deploying a real-time Q&A chatbot powered by semantic search and LLMs. It is suitable for applications requiring intelligent document-based question answering.
