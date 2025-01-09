import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import * 
from docprocess20 import load_env_variables, retrieve_relevant_documents  
from streamlit_chat import message  

# Load environment variables
load_env_variables()

# Initialize global variables for LLM and vector store
groq_api_key = os.getenv('groq_api_key')
llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma2-9b-it', temperature=0.1)
index_name = 'testproject4'

st.title("Langchain-QAbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I assist you?']

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template=""" Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I do not know.' """)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory,
                                 prompt=prompt_template,
                                 llm=llm,
                                 verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()

# User input for querying the vector store
with textcontainer:
     query = st.text_input("Query: ", key="input")
     if query:
         with st.spinner("typing..."):
             # Call retrieve_relevant_documents instead of get_similar_docs
             response_docs = retrieve_relevant_documents(query)  
            
             if response_docs:
                 combined_content = "\n".join([doc.page_content for doc in response_docs])
                 llm_response = llm.invoke(f"Based on the {combined_content} and {query} given from the user, give me the result only related to {query}. Remove unnecessary information from the {combined_content} which is irrelevant to {query}.")
                 response = llm_response.content.strip()
             else:
                 response = "No relevant information found."

             # Store user query and response in session state
             st.session_state.requests.append(query)
             st.session_state.responses.append(response)

# Display chat history
with response_container:
   if st.session_state['responses']:
       for i in range(len(st.session_state['responses'])):
           message(st.session_state['responses'][i], key=str(i))
           if i < len(st.session_state['requests']):
               message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
