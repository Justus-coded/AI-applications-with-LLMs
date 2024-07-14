# Data Roles Company Finder Chatbot

## Overview

This Streamlit app helps users find companies that have hired for data roles in the past. The dataset used consists of company names, job titles, salary estimates, company ratings, job descriptions, and company locations. Through this app, users can easily access this information using a single prompt.

## Datasets

The data were obtained from Kaggle and include the following datasets:
- [Data Analyst Jobs](https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs)
- [Business Analyst Jobs](https://www.kaggle.com/datasets/andrewmvd/business-analyst-jobs)
- [Data Engineer Jobs](https://www.kaggle.com/datasets/andrewmvd/data-engineer-jobs)
- [Data Scientist Jobs](https://www.kaggle.com/datasets/andrewmvd/data-scientist-jobs)

These datasets were preprocessed and combined into a single dataset to provide comprehensive information about various data roles.

## Tools Used

- **LangChain**: Used to build the LLM pipeline.
- **OpenAI API**: Used for text generation with the model `gpt-3.5-turbo`.
- **ChromaDB**: Used for the vector store.
- **all-MiniLM-L6-v2**: Used for text embeddings from the Hugging Face hub.
- **Streamlit**: Used to deploy the app.
- **Hugging Face Spaces**: Used to host and deploy the app.

## Building the App

### Step 1: Import Data and Prepare Vector Store

The combined dataset was loaded into a DataFrame and split into smaller chunks. The `all-MiniLM-L6-v2` model from Hugging Face was used to create text embeddings, which were then stored in ChromaDB. The ChromaDB directory was compressed into a zip file for easy loading later.

### Step 2: Loading the Vector Database

The app includes a function to load the vector database by extracting the ChromaDB zip file and loading it into memory with the appropriate embedding function.

### Step 3: Augmenting the User Prompt

To provide comprehensive responses, the app augments the user's query by adding relevant context from the vector store. This ensures that the responses are based on the most relevant information available.

### Step 4: Handling Chat with OpenAI

The augmented prompt is sent to OpenAI's `gpt-3.5-turbo` model to generate a response. The app handles the interaction with the OpenAI API and returns the generated text.

### Step 5: Streamlit UI

The Streamlit UI initializes the app, loads the vector store, and manages the chat interface. It displays the chat history and handles new user inputs, sending them to the LLM for responses.

## Setting Up OpenAI API Key

To use the OpenAI API, add your API key as a secret in the Hugging Face Spaces settings:

1. Go to your Hugging Face Space.
2. Navigate to the "Settings" tab.
3. Add your OpenAI API key as a secret with the name `OPENAI_API_KEY`.



## Deployment on Hugging Face Spaces

This app is deployed on Hugging Face Spaces using Streamlit. You can access the live app [here](https://huggingface.co/spaces/JustusI/data_roles). 

