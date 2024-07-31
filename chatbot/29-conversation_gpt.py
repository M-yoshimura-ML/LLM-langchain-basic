## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API tokens
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDF's and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")


def initialize_llm(api_key):
    """Initialize the Language Learning Model with the Groq API key."""
    return ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")


def upload_and_process_files(uploaded_files):
    """Process uploaded PDF files and return the extracted documents."""
    documents = []
    for uploaded_file in uploaded_files:
        print('uploaded_file:', uploaded_file)
        temppdf = f"./{uploaded_file.name}"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.read())  # Corrected the method used for reading the file
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
        os.remove(temppdf)  # Clean up the temporary file
    return documents


def create_vectorstore(documents, embeddings):
    """Create a vectorstore from the documents and embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def setup_chain(vectorstore, llm):
    """Set up the RAG chain with a retriever and a question-answering prompt."""
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize chat message history for a session."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def handle_user_input(rag_chain, session_id):
    """Process the user input and generate a response."""
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)


# Main logic
if api_key:
    llm = initialize_llm(api_key)

    # Manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    session_id = st.text_input("Session ID", value="default_session")
    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = upload_and_process_files(uploaded_files)
        vectorstore = create_vectorstore(documents, embeddings)
        rag_chain = setup_chain(vectorstore, llm)
        handle_user_input(rag_chain, session_id)

else:
    st.warning("Please enter the Groq API Key")
