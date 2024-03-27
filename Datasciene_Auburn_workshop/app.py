from openai import OpenAI
import streamlit as st
import os
import sys
from dotenv import load_dotenv, dotenv_values
load_dotenv()


# initialize the client
client = OpenAI(
  base_url="https://api-inference.huggingface.co/v1",
  api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')  #"hf_xxx" # Replace with your token
) 


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Google Gemma")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] #[{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for messasge in st.session_state.messages:
    st.chat_message(messasge["role"]).write(messasge["content"])

# React to user input
if prompt := st.chat_input():
    
     # Display user message in chat message container
    st.chat_message("user").write(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    ##Get response to the message using client
    response = client.chat.completions.create(model="google/gemma-2b-it", messages=st.session_state.messages)

    msg = response.choices[0].message.content
    
     # Display assistant response in chat message container
    st.chat_message("assistant").write(msg)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})