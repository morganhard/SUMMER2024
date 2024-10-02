import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain, load_csv_chat_chain
from streamlit_mic_recorder import mic_recorder,speech_to_text

from utils import get_timestamp, load_config, get_avatar
from image_handler import handle_image
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from csv_handler import handle_csv, handle_csv_chat, add_csv_documents_to_db
from langchain_experimental.agents.agent_toolkits import create_csv_agent

#from html_templates import css
from database_operations import load_last_k_text_messages, save_text_message, save_image_message, save_csv_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history, init_db
import sqlite3
config = load_config()

@st.cache_resource
def load_chain(usernamee, passwordd):
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain(st.session_state.selected_model, st.session_state.selected_temperature, usernamee, passwordd)
    elif st.session_state.csv_chat:
        print("loading csv chat")
        return load_csv_chat_chain(st.session_state.selected_model, st.session_state.selected_temperature, usernamee, passwordd)
    else:
        return load_normal_chain(st.session_state.selected_model, st.session_state.selected_temperature, usernamee, passwordd)

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def toggle_csv_chat():
    st.session_state.csv_chat = True
    clear_cache

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def main():
    init_db() 
    st.title("Multimodal")
    #st.write(css, unsafe_allow_html=True)
    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
        st.session_state.csv_uploader_key = 2
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    #st.session_state.csv_data = None
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    pdf_toggle_col, csv_toggle_col, voice_rec_col = st.sidebar.columns(3)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    csv_toggle_col.toggle("CSV Chat", key="csv_chat", value=False, on_change=clear_cache)
    with voice_rec_col:
        voice_recording=mic_recorder(start_prompt="Record Audio",stop_prompt="Stop recording", just_once=True)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)
    
    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")
    
    
    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"], key=st.session_state.audio_uploader_key)
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, 
                                            key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat)
    uploaded_csv = st.sidebar.file_uploader("Upload a csv file (not working yet)", type=["csv"], key=st.session_state.csv_uploader_key, on_change=toggle_csv_chat)#accept_multiple_files=True,
                                            

    st.sidebar.selectbox("Select model:", ["GPT4o", "Llama 3 70B"], key= "selected_model")
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,  # Minimum temperature
        max_value=2.0,  # Maximum temperature
        value=1.2,      # Default temperature
        step=0.1,       # Increment by 0.1
        format="%.1f",  # Display with one decimal place
        key="selected_temperature"
    )
    usernamee = st.sidebar.text_input("Infineon username:", key='access_username')
    passwordd = st.sidebar.text_input("Infineon password:", type='password',key='access_password')

    if uploaded_pdf:
        with st.spinner("Processing pdf..."):
            add_documents_to_db(uploaded_pdf)
            st.session_state.pdf_uploader_key += 2
    
    #if uploaded_csv:
    #    with st.spinner("Processing CSV..."):
    #        csv_data = uploaded_csv.read().decode('utf-8')
    #        df=handle_csv(csv_data)
    #        if df is not None:
    #            st.session_state.csv_data=df
    #            st.session_state.csv_uploader_key +=2
    #        #st.session_state.csv_chat = True


    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain(usernamee, passwordd)
        llm_answer = llm_chain.run(user_input = "Summarize this text: " + transcribed_audio, chat_history=[])
        save_audio_message(get_session_key(), "human", uploaded_audio.getvalue())
        save_text_message(get_session_key(), "ai", llm_answer)
        st.session_state.audio_uploader_key += 2

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain(usernamee, passwordd)
        llm_answer = llm_chain.run(user_input = transcribed_audio, 
                                   chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_audio_message(get_session_key(), "human", voice_recording["bytes"])
        save_text_message(get_session_key(), "ai", llm_answer)

    if uploaded_csv:
        with st.spinner("Processing csv..."):
            add_csv_documents_to_db(uploaded_csv)
            st.session_state.csv_uploader_key += 2
                #llm_chain = load_chain(usernamee,passwordd)

                #csv_data = uploaded_csv.read().decode('utf-8')
                #df=handle_csv(csv_data)
                #llm_chain = load_chain(usernamee,passwordd)
            
                #llm_answer = handle_csv_chat(uploaded_csv, user_input, st.session_state.selected_model, st.session_state.selected_temperature, usernamee, passwordd)
                #llm_chain = load_csv_chat_chain(st.session_state.selected_model, st.session_state.selected_temperature, usernamee, passwordd)
                #agent_executor = create_csv_agent(llm_chain, uploaded_csv, verbose=False)
                #response = agent_executor.run(user_input)
                #save_text_message(get_session_key(), "human", user_input)
                #save_csv_message(get_session_key(), "human", csv_data)
                #save_text_message(get_session_key(), "ai", llm_answer)
                #user_input = None
    
    if user_input and usernamee and passwordd:

        if uploaded_image:
            with st.spinner("Processing image..."):
                image_bytes = uploaded_image.read()
                llm_answer = handle_image(image_bytes, user_input)
                save_text_message(get_session_key(), "human", user_input)
                save_image_message(get_session_key(), "human", image_bytes)
                save_text_message(get_session_key(), "ai", llm_answer)
                user_input = None

        if user_input:
            llm_chain = load_chain(usernamee,passwordd)
            llm_answer = llm_chain.run(user_input = user_input, 
                                       chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
            save_text_message(get_session_key(), "human", user_input)
            save_text_message(get_session_key(), "ai", llm_answer)
            user_input = None


    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")

        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()
