# Citizen Science Resource Helper using LangGraph, VertexAI, and Qdrant

# -- Imports --
from dotenv import load_dotenv
import os
import uuid
import json
import pandas as pd
from typing import TypedDict, List

import vertexai
from qdrant_client import QdrantClient
from langchain_google_vertexai import ChatVertexAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import streamlit as st

load_dotenv()

# This is optional, but if you want to use Langsmith for tracing, you can set the following environment variables
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT_NAME")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize Vertex AI API (once per session, requires setup of google cloud authentication:https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
vertexai.init(project= os.getenv("GCLOUD_PROJECT_ID") , location= os.getenv("GCLOUD_REGION"))

# set up the LLM with the model name and parameters, make sure the model type is enabled in your Vertex AI project, quotas are set up, and the model is available in your region.
llm = ChatVertexAI(
    model_name="gemini-1.5-flash",
    temperature=0.25,
    max_output_tokens=2048
)

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
vector_store = QdrantVectorStore(
    client=client,
    collection_name= os.getenv("QDRANT_COLLECTION_NAME"),
    embedding=embeddings,
    content_payload_key = os.getenv("QDRANT_PAYLOAD_KEY")
)
content_retriever = vector_store.as_retriever()

# Metadata retriever (workaround for payload format issues). todo: Find a way to get the metadata in the same retriever call as the content
metadata_vector_store = QdrantVectorStore(
    client=client,
    collection_name= os.getenv("QDRANT_COLLECTION_NAME"),
    embedding=embeddings,
    content_payload_key = os.getenv("QDRANT_METADATA_KEY")
)
metadata_retriever = metadata_vector_store.as_retriever()

# -- Load link mapping CSV --
filename_to_link_df = pd.read_csv("filename_to_link.csv",delimiter= ";")

# -- Utility Functions to prepare the retrieved documents for readability by the llm--
def format_document(content_doc, metadata_doc):
    """Merges the retrieved content with relevant metadata."""
    if content_doc.metadata["_id"] == metadata_doc.metadata["_id"]:
        metadata_string = metadata_doc.page_content
        metadata_json = json.loads(metadata_string)
        metadata_dict = metadata_json["metadata"]
        filename = metadata_dict["filename"]
        languages = metadata_dict["languages"]
        title = filename.removesuffix(".pdf").removesuffix(".txt").replace("_"," ")
        link = filename_to_link_df[filename_to_link_df['filename'] == filename]['link'].values[0]
    else:
        title, page_number, link = "", "", "Not found"
    content = content_doc.page_content
    return f"Title: {title}\n Content: {content}\n Link: {link}\n"

def prepare_context(content_docs, metadata_docs, max_tokens=2500):
    """Format context and limit the tokens put into the context."""
    formatted_docs = []
    token_counter = 0
    for (content_doc, metadata_doc) in zip(content_docs, metadata_docs):
        formatted_doc = format_document(content_doc, metadata_doc)
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
        "Always cite the title and link provided in the context in your answer. \n"
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
    return {"retrieved_docs": formatted_context}

def generate_answer(state: RAGState) -> RAGState:
    question = state["question"]
    context = "\n".join(state["retrieved_docs"])
    prompt = rag_prompt.format(context=context,question=question)
    answer = llm.invoke(prompt)
    return {"answer": answer}

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
def start_conversation(question: str):
    thread_id = str(uuid.uuid4())  # Create a new unique thread ID for this conversation
    conversation_input = {"question": question}
    config = {"configurable": {"user_id": "1", "thread_id": thread_id}}
    
    stream = rag_graph.stream(conversation_input,config,stream_mode="values")
    responses = [response["answer"].content for response in stream if "answer" in response]
    return responses[-1] if responses else "No answer found."


# Streamlit UI
def main():
    st.title("Citizen Science Resource Helper")
    user_input = st.text_input(
    "Ask any question about how to do citizen science.\n this tool will search it's database for an appropriate document to answer your question."
    )

    if user_input:
        with st.spinner("Processing your question..."):
            answer = start_conversation(user_input)
        st.write("Answer: ", answer)

if __name__ == "__main__":
    main()