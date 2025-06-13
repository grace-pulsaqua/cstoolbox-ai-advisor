import streamlit as st

st.set_page_config(page_title="Citizen Science Resource Helper", page_icon=":robot_face:",layout="wide")

from app_functions import save_single_feedback_row, submit_message

import uuid
import os

# This is optional, but if you want to use Langsmith for tracing, you can set the following environment variables
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

#       with st.spinner("Loading the system... Please wait."): #Showing a loading spinner while initializing the external resources
#            try:
#                call_llm("test", st.session_state.user_id) # This is a dummy call to the LLM to initialize the connection and load the resources. It will not return any answer, but it will ensure that the LLM is loaded and ready to use.
#            except: # If there is an error while loading the external resources, show a user legible error message
#                st.error("There was a problem when connecting to the chat model. Please let us know if you encounter this error and we will try to fix it as soon as possble.")

#Create the streamlit user interface, accept user inputs
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
    🔬This tool is still being developed. If you encounter any problems or have ideas to make it better, please let us know at jonathanmeijers2000@gmail.com  
    '''
    st.markdown(instruction)
    
    disclaimer_text = '''
    This tool uses third-party services to process your questions and generate answers.
    You are subject to the data handling policies of the used service providers, Specifically:  
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
                            message["feedback_row_id"] = save_single_feedback_row(message) #saving the feedback to google sheets
                                      
if __name__ == "__main__":
    main()