## Impoting the dependencies
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

## Load the environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

## Loding the documents
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

## Setup the vectorstore
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200
        )
    
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

## Create the chains
def create_chains(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.2
    )
    
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        verbose = True
    )

    return chain

## Setup the app

st.set_page_config(
    page_title="Chat with Unstructured Documents",
    page_icon="📜📜📜",
    layout="centered"
)

st.title('🦙🦙Chat with Doc -- LLama 3.1 🦙🦙')


## Initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

## Upload the documents

upload_file = st.file_uploader(label="Upload your pdf file", type="pdf")

if upload_file:
    file_path = f'{working_dir}/{upload_file.name}'
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_documents(file_path))

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = create_chains(st.session_state.vectorstore)
     
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content']
                    )

user_input = st.chat_input('Ask a question')

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        response = st.session_state.conversational_chain({"question":user_input})
        assistant = response['answer']
        st.markdown(assistant)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant})