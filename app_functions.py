# This module contains the high-level functions for the Streamlit app, such as saving feedback to Google Sheets and submitting messages to the LLM.

#Downloaded packages
import streamlit as st

#Self-coded packages
from retrieve_external_resources import get_feedback_worksheet
from llm_functions import call_llm

#Built-in packages
import datetime

# --- SAVE FEEDBACK TO GOOGLE SHEETS --- saves a single row of feedback to the google sheet. 
def save_single_feedback_row(message):
    worksheet = get_feedback_worksheet()
    row_data = [ # takes all included data of the interaction with a timestamp
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.user_id,
        message["question"],
        message["answer"],
        message.get("feedback")
    ]
    
    if "row_id" in message:
        worksheet.update(f"A{message['row_id']}:E{message['row_id']}", [row_data]) # This only happens if feedback was already provided for that particular message. In this case, it updates the previous feedback.
        return message["row_id"]
    else:
        worksheet.append_row(row_data)
        return worksheet.row_count

#--- SUBMIT MESSAGE FUNCTION --- Implemented this way to avoid the need for a submit button, instead running the call whenever the user presses enter in the question box.
def submit_message():
    if st.session_state.question_box != "":
        with st.spinner("Processing your question: Searching the database and generating a response..."):
            answer = call_llm(st.session_state.question_box, st.session_state.user_id)
            st.session_state.chat_history.append({"question": st.session_state.question_box, "answer": answer}) # save the answers to the streamlit session state. This acts as front-end conversation memory, but it should be possible to integrate this with the LangGraph state.
            st.session_state.question_box = "" #clear the question box at the end to avoid repeating the llm call when the page is updated
