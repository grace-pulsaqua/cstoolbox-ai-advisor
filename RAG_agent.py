# Citizen Science Resource Helper using LangGraph, VertexAI, and Qdrant

# -- Imports --
import os
import uuid
import json
import pandas as pd
from typing import TypedDict, List
import streamlit as st
st.set_page_config(page_title="Citizen Science Resource Helper", page_icon=":robot_face:",layout="wide")

import vertexai
from google.oauth2 import service_account
from qdrant_client import QdrantClient
from langchain_google_vertexai import ChatVertexAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# This is optional, but if you want to use Langsmith for tracing, you can set the following environment variables
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


# set up the LLM with the model name and parameters, make sure the model type is enabled in your Vertex AI project, quotas are set up, and the model is available in your region.
@st.cache_resource
def load_llm():
    credentials_json = dict(st.secrets["gcloud"]['my_project_settings'])
    credentials_json["private_key"] = credentials_json["private_key"].replace(",", "\n")
    credentials = service_account.Credentials.from_service_account_info(credentials_json)
    # Initialize Vertex AI API when you have downloaded your service account credentials JSON file and entered it in the secrets.toml file: https://discuss.streamlit.io/t/how-to-use-an-entire-json-file-in-the-secrets-app-settings-when-deploying-on-the-community-cloud/49375/2
    vertexai.init(project= os.getenv("GCLOUD_PROJECT_ID") , location= os.getenv("GCLOUD_REGION"),credentials= credentials)
    
    return ChatVertexAI(
        model_name="gemini-1.5-flash",
        temperature=0.25,
        max_output_tokens=3100
        )

#load the vector store retrievers
@st.cache_resource
def load_vector_stores():
    #configure the Azure OpenAI embeddings model
    embeddings = AzureOpenAIEmbeddings(
        deployment= os.getenv("AZURE_DEPLOYMENT_NAME"),
        model= os.getenv("AZURE_MODEL_NAME"), 
        azure_endpoint= os.getenv("AZURE_ENDPOINT"), 
        openai_api_type="azure",
        openai_api_version= os.getenv("AZURE_API_VERSION"), 
        openai_api_key= os.getenv("AZURE_OPENAI_API_KEY"), 
    )
    
    #Configure the Qdrant client and endpoint for connecting to the vector stores
    client = QdrantClient(
        url= os.getenv("QDRANT_URL"), 
        api_key = os.getenv("QDRANT_API_KEY")
    )

    # Primary content retriever
    content_store = QdrantVectorStore(
        client=client,
        collection_name= os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
        content_payload_key = os.getenv("QDRANT_PAYLOAD_KEY")
    )
    # Metadata retriever (workaround for payload format issues). todo: Find a way to get the metadata in the same retriever call as the content
    metadata_vector_store = QdrantVectorStore(
    client=client,
    collection_name= os.getenv("QDRANT_COLLECTION_NAME"),
    embedding=embeddings,
    content_payload_key = os.getenv("QDRANT_METADATA_KEY")
    )
    return content_store.as_retriever(search_kwargs = {"k": 3}), metadata_vector_store.as_retriever(search_kwargs = {"k": 3})

# -- Load link mapping CSV --
@st.cache_data
def load_links_to_data_files():
    return pd.read_csv("links_to_data_files.csv", delimiter=";")

with st.spinner("Loading the system... Please wait."):
    llm = load_llm()
    content_retriever, metadata_retriever = load_vector_stores()
    links_to_data_files_df = load_links_to_data_files()
    
# -- Utility Functions to prepare the retrieved documents for readability by the llm--

def prepare_context(content_docs, metadata_docs, max_tokens=1500):
    """Format context and limit the amount of words put into the context."""
    formatted_docs = []
    token_counter = 0
    
    for (content_doc, metadata_doc) in zip(content_docs, metadata_docs):
        #match content and metadata documents by their IDs, otherwise use empty metadata
        if content_doc.metadata["_id"] == metadata_doc.metadata["_id"]:
            try:
                metadata = json.loads(metadata_doc.page_content)["metadata"]
            except (json.JSONDecodeError, KeyError, TypeError):
                metadata = {}
        else:
            metadata = {}
        
        filename = metadata.get("filename", "")
        title = filename.removesuffix(".pdf").removesuffix(".txt").replace("_"," ")
        link_row = links_to_data_files_df[links_to_data_files_df['filename'] == filename]
        link = link_row['link'].values[0] if not link_row.empty else "Not found"
        content = content_doc.page_content
        
        formatted_doc = (f"Title: {title}\n Content: {content}\n Link: {link}")
        
        doc_tokens = len(formatted_doc.split())  # Simple word count as proxy
        if token_counter + doc_tokens > max_tokens:
            break  # Stop adding more docs once near token limit
        formatted_docs.append(formatted_doc)
        token_counter += doc_tokens
    return formatted_docs

# -- Prompt Template --
rag_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a database helper for answering questions about citizen science methods, tools, and best practices based on a database of resources about citizen science.\n"
        "You will be provided relevant context from the database to help you answer. \n"
        "The context includes the title of the original document and a link to that document \n"
        "If the answer to the question is not found in the context, say you don't know the answer and offer to guess, explicitly marking guesses.\n"
        "Cite each context document that you used by providing the title and link at the bottom of your answer in a separate line for each document. Do not repeat duplicate document titles or links. \n"
        "Answer in the same language as the question.\n"
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n \nQuestion:\n{question}"
    )
    
])

# RAG agent structure
class RAGState(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str

# -- Node Functions --
def retrieve_docs(state: RAGState) -> RAGState:
    question = state["question"]
    content_docs = content_retriever.invoke(question)
    metadata_docs = metadata_retriever.invoke(question)
    formatted_context = prepare_context(content_docs, metadata_docs)
    state["retrieved_docs"] = formatted_context  # Update state with retrieved docs
    return state    

def generate_answer(state: RAGState) -> RAGState:
    question = state["question"]
    context = "\n".join(state["retrieved_docs"])
    prompt = rag_prompt.format(context=context,question=question)
    answer = llm.invoke(prompt)
    state["answer"] = answer  # Update state with the generated answer
    return state

# -- Build the LangGraph --
memory = MemorySaver()
graph = StateGraph(RAGState)

# Register nodes
graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

# Define state transitions
graph.add_edge(START,"retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)  # End of one cycle

# Compile the graph - todo: figure out how to make the memory functional
rag_graph = graph.compile(checkpointer=memory)

# -- Conversation function --
def call_llm(question: str, thread_id: str):
    conversation_input = {"question": question}
    config = {"configurable": {"user_id": "1", "thread_id": thread_id}}
    stream = rag_graph.stream(conversation_input,config,stream_mode="values")
    responses = [response["answer"].content for response in stream if "answer" in response]
    return responses[-1] if responses else "Model error: Unable to return an answer. If you encounter this error, please contact the developer."

# Streamlit UI
def main():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4()) # Create a new unique thread ID for this conversation session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("Citizen Science Resource Helper")
    instruction = '''🛠️This tool helps you find information about citizen science methods, tools, and best practices.  
    🔎For example, try asking it questions about how to setup a water quality monitoring initiative, how to find participants for your activity, or what projects already exist for monitoring biodiversity.  
    📄It will search a database of curated documents for an answer to your question and try to answer based on that.  
    😞Unfortunately, the chat model does not have any memory right now, so it will not remember what your previous question was. Give as much detail as possible for every question.  
    ⁉️If you have any questions or feedback, please contact the developer at jonathan.stage@pulsaqua.nl'''
    st.markdown(instruction)
    
    user_input = st.text_input(
    "Ask your question here:"
    )

    if user_input:
        with st.spinner("Processing your question: Searching the database and generating a response..."):
            answer = call_llm(user_input, st.session_state.user_id)
        st.session_state.chat_history.append({"question": user_input, "answer": answer})
        
    if st.session_state.chat_history:
        st.markdown("**Conversation History**")
        for chat in reversed(st.session_state.chat_history):
            with st.container():
                st.write(f":gray-background[🧑‍💻You:]\n{chat['question']}")
                st.write(f":gray-background[🤖 Helper:]\n{chat['answer']}")

if __name__ == "__main__":
    main()