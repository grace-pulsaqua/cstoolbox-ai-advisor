import os
import uuid
import json
import tempfile
import pandas as pd
from typing import TypedDict, List
import streamlit as st
st.set_page_config(page_title="Citizen Science Resource Helper", page_icon=":robot_face:",layout="wide")
from google.oauth2 import service_account
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import gspread
from gspread_dataframe import set_with_dataframe
import datetime

# This is optional, but if you want to use Langsmith for tracing, you can set the following environment variables
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize Google cloud credentials when you have downloaded your service account credentials JSON file and entered it in the secrets.toml file: https://discuss.streamlit.io/t/how-to-use-an-entire-json-file-in-the-secrets-app-settings-when-deploying-on-the-community-cloud/49375/2
@st.cache_resource
def get_gcloud_credentials(scopes):
    creds_dict = dict(st.secrets["gcloud"]["my_project_settings"])
    creds_dict["private_key"] = creds_dict["private_key"].replace(",", "\n")

    #this way is necessary to circumvent google auth trying to access an environment variable that is not set in the streamlit cloud
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(creds_dict, f)
        temp_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_dict)
    return service_account.Credentials.from_service_account_file(temp_path, scopes=scopes)

@st.cache_resource
def load_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    # set up the LLM with the model name and parameters, make sure that you have an api key set in google ai studio: https://aistudio.google.com/apikey
    return ChatGoogleGenerativeAI(
        google_api_key = st.secrets["GOOGLE_GENAI_API_KEY"],
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_output_tokens=1024
        )

# Optional function to connect to a google sheet for feedback logging
@st.cache_resource
def get_feedback_worksheet():
    credentials = get_gcloud_credentials(["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key(st.secrets["GSHEET_FEEDBACK_KEY"])
    try:
        return sh.worksheet("Feedback")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title="Feedback", rows=1000, cols=20)
        set_with_dataframe(worksheet, pd.DataFrame(columns=["Timestamp", "User ID", "Question", "Answer", "Feedback"]))
        return worksheet

#load the vector store retrievers, make sure you have setup the correct environment variables in the secrets.toml file: https://docs.streamlit.io/streamlit-cloud/deploy/create-secrets
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

with st.spinner("Loading the system... Please wait."):
    try:
        content_retriever, metadata_retriever = load_vector_stores()
        links_to_data_files_df = load_links_to_data_files()
        llm = load_llm()
    except:
        st.error("There was a problem when connecting to the chat model. We are aware of this problem, and are working on a solution :) Please try again later.")
    
def prepare_context(content_docs, metadata_docs, max_tokens=1500):
    """Format context and limit the amount of words put into the context."""
    formatted_docs = []
    token_counter = 0
    
    for (content_doc, metadata_doc) in zip(content_docs, metadata_docs):
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
        
        formatted_doc = (f"Title: {title}\n Content: {content}\n Link: {link}\n")
        
        doc_tokens = len(formatted_doc.split())  
        if token_counter + doc_tokens > max_tokens:
            break  
        formatted_docs.append(formatted_doc)
        token_counter += doc_tokens
    return formatted_docs

rag_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a database helper for answering questions about citizen science methods, tools, and best practices using a database of resources about citizen science.\n"
        "You will be provided relevant context from the database to help you answer. \n"
        "Try to give an answer of at least 150 words with concrete examples.\n"
        "If the answer to the question is not found in the context, answer with your best guess but say that you are guessing.\n"
        "Always provide at least 1 practical action that the user could take to get more information about their question.\n"
        "Aways suggest at least 1 follow-up question that the user could ask you to get more information.\n"
        "The context includes the title of the original document and a link to that document \n"
        "Cite each context document that you used by providing the title and link at the bottom of your answer in a separate line for each document. Do not repeat duplicate document titles or links. \n"
        "The question can be asked in many different languages. Give your answer in the same language as the question and translate relevant context if it is presented in another language than the question.\n"
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n \nQuestion:\n{question}"
    )
    
])

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str

def retrieve_docs(state: RAGState) -> RAGState:
    question = state["question"]
    content_docs = content_retriever.invoke(question)
    metadata_docs = metadata_retriever.invoke(question)
    formatted_context = prepare_context(content_docs, metadata_docs)
    state["retrieved_docs"] = formatted_context  
    return state    

def generate_answer(state: RAGState) -> RAGState:
    question = state["question"]
    context = "\n".join(state["retrieved_docs"])
    prompt = rag_prompt.format(context=context,question=question)
    answer = llm.invoke(prompt)
    state["answer"] = answer  
    return state

memory = MemorySaver()
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.add_edge(START,"retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)  # End of one cycle

# todo: figure out how to make the memory functional
rag_graph = graph.compile(checkpointer=memory)

def call_llm(question: str, thread_id: str):
    conversation_input = {"question": question}
    config = {"configurable": {"user_id": "1", "thread_id": thread_id}}
    stream = rag_graph.stream(conversation_input,config,stream_mode="values")
    responses = [response["answer"].content for response in stream if "answer" in response]
    return responses[-1] if responses else " Model error: We were able to connect to the model, but it did not generate an answer. Try rephrasing your question, if that doesn't work, the code is likely broken in a fundamental way. Please let us know if this is the case."

def save_single_feedback_row(message):
    worksheet = get_feedback_worksheet()
    row_data = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.user_id,
        message["question"],
        message["answer"],
        message.get("feedback")
    ]
    
    if "row_id" in message:
        worksheet.update(f"A{message['row_id']}:E{message['row_id']}", [row_data])
        return message["row_id"]
    else:
        worksheet.append_row(row_data)
        return worksheet.row_count

def submit_message():
    if st.session_state.question_box != "":
        with st.spinner("Processing your question: Searching the database and generating a response..."):
            answer = call_llm(st.session_state.question_box, st.session_state.user_id)
            st.session_state.chat_history.append({"question": st.session_state.question_box, "answer": answer})
            st.session_state.question_box = ""

def main():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4()) 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("🤖Citizen Science Resource Helper🤖")
    instruction = '''🛠️This tool helps you find information about methods, tools, and best practices for water-related citizen science.  
    🔎For example, try asking it questions about how to setup a water quality monitoring initiative, how to find participants for your activity, or what projects already exist for monitoring biodiversity.  
    🧠It will search a database of curated documents for an answer to your question. Links to the documents will be provided in the answer.  
    📝For a list of documents in the database, check https://github.com/J-na/CS_advisor/blob/main/links_to_data_files.csv  
    🔠The model works best in English, but it can understand many other European languages if you feel more comfortable asking questions in your native tongue!  
    😞Unfortunately the chat model does not have any memory right now, so it will not remember what your previous question was. Give as much detail as possible for every question.  
    '''
    st.markdown(instruction)
    
    disclaimer_text = '''
    This tool uses third-party services to process your questions and generate answers.
    By using this app, you are subject to the data handling policies of the mentioned service providers, Specifically:  
        Google GenAI is used to generate responses via their Gemini-2.0-flash-lite model: https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance
        The OpenAI text-embedding-3-model is used to embed your questions for searching a vector database: https://openai.com/policies/data-processing-addendum/
    
    Submitted questions, AI-generated answers, and optional feedback are stored for 14 days for debugging and quality improvement.  
    No personally identifiable information is collected unless you included it in your question.
    
    For more information, contact the developer at jonathanmeijers2000@gmail.com  
    '''
    
    with st.expander("📜 Disclaimer"):
        st.markdown(disclaimer_text)

    st.text_input("Ask your question here:", value="", key = "question_box", on_change= submit_message)

    if st.session_state.chat_history:
        st.markdown("**Conversation History**")
        for message in reversed(st.session_state.chat_history):
            st.write(f"🧑‍💻 **You:** {message['question']}")
            st.write(f"🤖 **Helper:** {message['answer']}")
            
            if "feedback" not in message:
                with st.container(border= True):
                    st.write("How useful was this answer?")
                    feedback = st.feedback("stars", key=f"feedback_{hash(message['question'])}")
                    if feedback:
                        with st.spinner("Thank you for your feedback! Sending to database ..."):
                            message["feedback"] = feedback
                            message["feedback_row_id"] = save_single_feedback_row(message)
                                      
if __name__ == "__main__":
    main()