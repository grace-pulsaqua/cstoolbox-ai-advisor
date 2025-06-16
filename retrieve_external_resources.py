# This module contains all the functions needed to connect to the external API resources such as the vector store, the LLM, and the google sheets for feedback logging.

import streamlit as st
from google.oauth2 import service_account
import gspread
from gspread_dataframe import set_with_dataframe
import pandas as pd

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # We use this instead of VertexAI because the vertexAI package requires grcpio version 1.91 which conflicts with the google sheets package requiring grcpio 1.71. For the purpose of this app there is little difference between the GenAI and vertexAI packages.

import tempfile
import json
import os

# --- CONNECT TO GOOGLE CLOUD SERVICES --- Initialize Google cloud credentials for accessing google sheets for feedback logging or using vertex AI. This is not necessary when using the google GenAI package to connect to an LLM.
@st.cache_resource
def get_gcloud_credentials(scopes):
    creds_dict = dict(st.secrets["gcloud"]["my_project_settings"]) # Make sure your service account JSON is stored in the secrets.toml file under the key "gcloud" and the subkey "my_project_settings"
    creds_dict["private_key"] = creds_dict["private_key"].replace(",", "\n") # convert the toml format back to valid JSON

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(creds_dict, f)
        temp_path = f.name
    #Load the credentials into an environment variable to prevent an issue where google auth tries to access the environment variable instead of using the file path in the function call
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_dict)
    return service_account.Credentials.from_service_account_file(temp_path, scopes=scopes) # Return a service account object for later use in authentication

# --- CONNECT TO THE LLM ---
@st.cache_resource
def load_llm():
    # set up the LLM with the model name and parameters, 
    return ChatGoogleGenerativeAI(
        google_api_key = st.secrets["GOOGLE_GENAI_API_KEY"], # The api key set in google ai studio: https://aistudio.google.com/apikey
        model="gemini-2.5-flash-preview-05-20", # The model to use, this is the most cost-effective model that still gives good results. You can also use "gemini-2.0-pro" for better results, but it is more expensive.
        temperature=0, #Temperature controls the randomness of the model's output. 0 means it will always give the same answer to the same question, 1 means it will be more creative and varied.
        max_output_tokens=2048 # Maximum number of tokens in the output, this is mainly to limit the cost of the API call. It can be increased to allow for more elaborate responses.
        )

# --- GOOGLE SHEET CONNECTION --- 
# Optional function to connect to a google sheet for feedback logging 
@st.cache_resource
def get_feedback_worksheet():
    credentials = get_gcloud_credentials(["https://www.googleapis.com/auth/spreadsheets"]) # specify the spreadsheets scope
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key(st.secrets["GSHEET_FEEDBACK_KEY"]) #Make sure your google sheet key is stored in the secrets.toml file 
    try:
        return sh.worksheet("Feedback")
    except gspread.exceptions.WorksheetNotFound: #creates a feedback worksheet if it does not exist yet
        worksheet = sh.add_worksheet(title="Feedback", rows=1000, cols=20)
        set_with_dataframe(worksheet, pd.DataFrame(columns=["Timestamp", "User ID", "Question", "Answer", "Feedback"]))
        return worksheet

# --- LOAD VECTOR STORE RETRIEVERS --- 
@st.cache_resource
def load_vector_stores():
    
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key= st.secrets["OPENAI_API_KEY"],
    )
    
    #uncomment this block if you want to use azure openai embeddings instead of the default openai embeddings
    #embeddings = AzureOpenAIEmbeddings(
    #    deployment= st.secrets["AZURE_DEPLOYMENT_NAME"],
    #    model= st.secrets["AZURE_MODEL_NAME"], 
    #    azure_endpoint= st.secrets["AZURE_ENDPOINT"], 
    #    openai_api_type="azure",
    #    openai_api_version= st.secrets["AZURE_API_VERSION"], 
    #    openai_api_key= st.secrets["AZURE_OPENAI_API_KEY"], 
    #)
        
    client = QdrantClient(
        url= st.secrets["QDRANT_URL"], 
        api_key = st.secrets["QDRANT_API_KEY"]
    )

    # Primary content retriever
    content_store = QdrantVectorStore(
        client=client,
        collection_name= st.secrets["QDRANT_COLLECTION_NAME"],
        embedding=embeddings,
        content_payload_key = st.secrets["QDRANT_PAYLOAD_KEY"]
    )
    
    # Metadata retriever (workaround for payload format issues). todo: Find a way to get the metadata in the same retriever call as the content
    metadata_vector_store = QdrantVectorStore(
    client=client,
    collection_name= st.secrets["QDRANT_COLLECTION_NAME"],
    embedding=embeddings,
    content_payload_key = st.secrets["QDRANT_METADATA_KEY"]
    )
    return content_store.as_retriever(search_kwargs = {"k": 3}), metadata_vector_store.as_retriever(search_kwargs = {"k": 3})

# -- Load link mapping CSV -- This is a manually created file with an online link to each document in the database based on filename. As of 01/05/2025 all the links were live.
@st.cache_data
def load_links_to_data_files():
    return pd.read_csv("links_to_data_files.csv", delimiter=",")

