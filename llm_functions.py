#This module contains the functions and setup for the RAG system itself

#Downloaded packages
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

#Self-coded packages
from retrieve_external_resources import load_links_to_data_files,load_vector_stores,load_llm

#Built-in packages
import json
from typing import TypedDict, List

# Get the vector store retrievers and the LLM
content_retriever, metadata_retriever = load_vector_stores()
llm = load_llm()

# --- PROMPT TEMPLATE ---Implemented with LangChain to enable LangGraph tracking and easy distinction between system prompt and user prompt.
rag_prompt = ChatPromptTemplate.from_messages([  
    SystemMessagePromptTemplate.from_template(
        "You are a database helper for answering questions about citizen science methods, tools, and best practices using a database of resources about citizen science.\n"
        "You will be provided relevant documents from the database to help you answer. \n"
        "Give an answer of at least 200 words with concrete examples.\n"
        "If the answer to the question is not found in the documents, answer with your best guess but say that you are guessing.\n"
        "Always provide at least 1 practical action that the user could take to get more information about their question.\n"
        "Aways suggest at least 1 follow-up question that the user could ask you to get more information.\n"
        "The provided documents include a title and link to that document \n"
        "Cite each document that you used by providing the title and link at the bottom of your answer in a separate line for each document. Do not repeat duplicate document titles or links. \n"
        "The question can be asked in many different languages. Give your answer in the same language as the question and translate relevant context if it is presented in another language than the question.\n"
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n \nQuestion:\n{question}"
    )
])

# --- LLM GRAPH SETUP ---
class RAGState(TypedDict): #Define the properties of the RAG system (these will be used as the state of the graph)
    question: str
    retrieved_docs: List[str]
    answer: str

def retrieve_docs(state: RAGState) -> RAGState: #Define the function to retrieve documents based on the question
    question = state["question"]
    content_docs = content_retriever.invoke(question) # Retrieve relevant content based on the question
    metadata_docs = metadata_retriever.invoke(question) # Retrieve relevant metadata based on the question
    formatted_context = prepare_context(content_docs, metadata_docs) # Format the retrieved documents into a context string for the LLM
    state["retrieved_docs"] = formatted_context  
    return state # Update the state class with the retrieved documents

def generate_answer(state: RAGState) -> RAGState: # Define the function to generate an answer based on the retrieved documents and question
    question = state["question"]
    context = "\n".join(state["retrieved_docs"])
    prompt = rag_prompt.format(context=context,question=question) #Format the prompt based on the predefined template and the retrieved context and question
    answer = llm.invoke(prompt)
    state["answer"] = answer  
    return state # Update the state class with the obtained answer

memory = MemorySaver()
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.add_edge(START,"retrieve") #The graph starts by always retrieving documents
graph.add_edge("retrieve", "generate") #Afterwards, it generates an answer based on the retrieved documents
graph.add_edge("generate", END) #Finally, it ends the graph after generating the answer

# todo: figure out how to make the memory functional, the checkpointer = memory argument does nothing in the current implementation
rag_graph = graph.compile(checkpointer=memory)

# --- FORMAT RETRIEVED CONTEXT ---
def prepare_context(content_docs, metadata_docs, max_tokens=1500):
    formatted_docs = []
    token_counter = 0 # This is incremented to keep track of the number of tokens in the formatted documents, to avoid incurring large API call costs.
    
    for (content_doc, metadata_doc) in zip(content_docs, metadata_docs):
        if content_doc.metadata["_id"] == metadata_doc.metadata["_id"]: #If the retrieved metadata doc is indeed the same as the content doc, we use that metadata. Otherwise it is empty
            try:
                metadata = json.loads(metadata_doc.page_content)["metadata"]
            except (json.JSONDecodeError, KeyError, TypeError):
                metadata = {}
        else:
            metadata = {}
        
        filename = metadata.get("filename", "")
        title = filename.removesuffix(".pdf").removesuffix(".txt").replace("_"," ") # Get the doc title from the filename in the metadata
        links_to_data_files_df = load_links_to_data_files()
        link_row = links_to_data_files_df[links_to_data_files_df['filename'] == filename] # Retrieve the link to the document by matching the filenmame in the metadata with the filename in the provided links_to_data_files csv
        link = link_row['link'].values[0] if not link_row.empty else "Not found" 
        content = content_doc.page_content
        
        formatted_doc = (f"Title: {title}\n Content: {content}\n Link: {link}\n") # insert the title, content, and link into a formatted string
        
        doc_tokens = len(formatted_doc.split())  
        if token_counter + doc_tokens > max_tokens:
            break  
        formatted_docs.append(formatted_doc)
        token_counter += doc_tokens
    return formatted_docs

# --- CALL LLM API ---
def call_llm(question: str, thread_id: str): #Takes a thread ID - This is not necessary for the current implementation, but it could be useful to track conversation history in the future.
    conversation_input = {"question": question}
    config = {"configurable": {"user_id": "1", "thread_id": thread_id}} # This config is necessary for the 'stream' mode of the graph. This is necessary for enabling conversation memory, but this doesn't work in the current implementation.
    stream = rag_graph.stream(conversation_input,config,stream_mode="values")
    responses = [response["answer"].content for response in stream if "answer" in response] #returns all responses from the graph stream, which are the answers generated by the LLM.
    return responses[-1] if responses else " Model error: We were able to connect to the model, but it did not generate an answer. Try rephrasing your question, if that doesn't work, the code is likely broken in a fundamental way. Please let us know if this is the case." # it returns the last response or a fatal error. This should never be encountered, because the code should fail at an earlier stage if there is an error in the model connection.
