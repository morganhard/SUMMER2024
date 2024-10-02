import pandas as pd
from io import StringIO
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.document_loaders import CSVLoader
from llm_chains import create_embeddings, csv_load_vectordb
from llm_chains import load_csv_chat_chain
from prompt_templates import csv_chat_prompt
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def add_csv_documents_to_db(csv_bytes):
    data = csv_loader(csv_bytes)
    documents = split_csv(data)
    vector_db = csv_load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
    print("Documents added to db.")


def csv_loader(csv_bytes):
    loader = CSVLoader(file_path=csv_bytes)
    data = loader.load()
    return data

def split_csv(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    return docs

#DO NOT NEED ANYMORE
def handle_csv(csv_data):
    """CSV file processing"""
    try:
        df = pd.read_csv(StringIO(csv_data))
        return df
    except Exception as e:
        st.error("Error processing CSV: {str(e)}")
        return None
    
#DO NOT NEED ANYMORE
def handle_csv_chat(uploaded_csv, user_input, model, temperature, username, password):
    """Interact with LLM"""
    #csv_string = df.to_csv(index=False)
    #message = user_query
    loader = CSVLoader(file_path=uploaded_csv)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, create_embeddings())
    llm_chain = load_csv_chat_chain(model, temperature, username, password)
    qa = ConversationalRetrievalChain.from_llm(
        llm_chain,
        db.as_retriever,
        return_source_documents = True
    )
    #result = qa({"question": user_input, "chat_history": chat_history})
    agent_executor = create_csv_agent(llm_chain, uploaded_csv, verbose=False)
    response = agent_executor.run(user_input)
    return response