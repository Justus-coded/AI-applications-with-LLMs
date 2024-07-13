import streamlit as st
import os
import zipfile
import pandas as pd
from langchain.document_loaders import DataFrameLoader
#import tiktoken
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
import requests

# Function to load vector database
def load_vector_db(zip_file_path, extract_path):
    with st.spinner("Loading vector store..."):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=extract_path,
        embedding_function=embedding_function
    )
    st.success("Vector store loaded")
    return vectordb

# Function to augment prompt
def augment_prompt(query, vectordb):
    results = vectordb.similarity_search(query, k=10)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""
    You are an AI assistant. Use the context provided below to answer the question as comprehensively as possible. 
    If the answer is not contained within the context, respond politely that you cannot provide that information.
    Context:
    {source_knowledge}
    Question: {query}
    """
    return augmented_prompt


# Function to handle chat with OpenAI
def chat_with_openai(query, vectordb, openai_api_key):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    augmented_query = augment_prompt(query, vectordb)
    prompt = HumanMessage(content=augmented_query)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        prompt
    ]
    res = chat(messages)
    return res.content


# Streamlit UI
st.title("Data Roles Company Finder Chatbot")
st.write("This app helps users find companies hiring for data roles, providing information such as job title, salary estimate, job description, company rating, and more.")

# Load vector database
zip_file_path = "chroma_db_compressed_.zip"
extract_path = "./chroma_db_extracted"
vectordb = load_vector_db(zip_file_path, extract_path)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        response = chat_with_openai(prompt, vectordb, openai_api_key)
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})