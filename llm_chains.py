from prompt_templates import memory_prompt_template, pdf_chat_prompt, csv_chat_prompt
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from operator import itemgetter
from utils import load_config
import chromadb


import base64
import httpx
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

config = load_config()

def generate_base64_string(username, password):
    sample_string = username + ":" + password
    sample_string_bytes = sample_string.encode("ascii")
    base64_bytes = base64.b64encode(sample_string_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string

def create_llm(temperature, modelname, username, password):
    #username=
    #password=
    #llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    if not username or not password:
        raise ValueError("USERNAME or PASSWORD not found")
    
    base64_string = generate_base64_string(username, password)
    
    llm = ChatOpenAI(
    model_name=modelname,
    openai_api_base='https://gpt4ifx.icp.infineon.com/',
    openai_api_key="EMPTY",  # Replace with your actual API key if needed
    default_headers={
        'Authorization': f"Basic {base64_string}"
    },
    http_client=httpx.Client(verify="./ca-bundle.crt"),  # Adjust if needed
    temperature=temperature,
    )
    return llm

#def create_embeddings(embeddings_path = config["embeddings_path"]):
#    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)
def create_embeddings(google_api_key = config["GOOGLE_API_KEY"]):
    return GoogleGenerativeAIEmbeddings(google_api_key=google_api_key,model="models/text-embedding-004")

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt):
    return LLMChain(llm=llm, prompt=chat_prompt)
    
def load_normal_chain(model_name, temperature, username, password):
    if model_name == "GPT4o":
        modelname='gpt-4o'
        return chatChain(temperature, modelname, username, password)  # Assuming chatChain is for GPT4 Turbo
    elif model_name == "Llama 3 70B":
        modelname='llama3-70b'
        return chatChain(temperature, modelname, username, password)  # Create a new class for Llama 3 70B
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    #return chatChain()

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )
    return langchain_chroma

#CSV CHANGES
def csv_load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["csv_chromadb"]["csv_chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["csv_chromadb"]["csv_collection_name"],
        embedding_function=embeddings,
    )
    return langchain_chroma

def load_pdf_chat_chain(model_name, temperature, username, password):
    if model_name == "GPT4o":
        modelname='gpt-4o'
        return pdfChatChain(temperature, modelname, username, password)  # Assuming chatChain is for GPT4 Turbo
    elif model_name == "Llama 3 70B":
        modelname='llama3-70b'
        return pdfChatChain(temperature, modelname, username, password)  # Create a new class for Llama 3 70B
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    #return pdfChatChain()

def load_csv_chat_chain(model_name, temperature, username, password):
    if model_name == "GPT4o":
        modelname='gpt-4o'
        return csvChatChain(temperature, modelname, username, password)  # Assuming chatChain is for GPT4 Turbo
    elif model_name == "Llama 3 70B":
        modelname='llama3-70b'
        return csvChatChain(temperature, modelname, username, password)  # Create a new class for Llama 3 70B
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_retrieval_chain(llm, vector_db):
    return RetrievalQA.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}), verbose=True)

def create_pdf_chat_runnable(llm, vector_db, prompt):
    runnable = (
        {
        "context": itemgetter("human_input") | vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}),
        "human_input": itemgetter("human_input"),
        "history" : itemgetter("history"),
        }
    | prompt | llm.bind(stop=["Human:"]) 
    )
    return runnable

def create_csv_chat_runnable(llm, vector_db, prompt):
    runnable = (
        {
        "context": itemgetter("human_input") | vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}),
        "human_input": itemgetter("human_input"),
        "history" : itemgetter("history"),
        }
    | prompt | llm.bind(stop=["Human:"]) 
    )
    return runnable

class pdfChatChain:

    def __init__(self, temperature, modelname, username, password):
        vector_db = load_vectordb(create_embeddings())
        llm = create_llm(temperature, modelname, username, password)
        #llm = load_ollama_model()
        prompt = create_prompt_from_template(pdf_chat_prompt)
        self.llm_chain = create_pdf_chat_runnable(llm, vector_db, prompt)

    def run(self, user_input, chat_history):
        print("Pdf chat chain is running...")
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history})
    
class csvChatChain:
    def __init__(self, temperature, modelname, username, password):
        vector_db = csv_load_vectordb(create_embeddings())
        llm = create_llm(temperature, modelname, username, password)
        prompt = create_prompt_from_template(csv_chat_prompt)
        self.llm_chain = create_csv_chat_runnable(llm, vector_db, prompt)

    def run(self, user_input, chat_history):
        print("CSV chat chain is running...")
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history})   


class chatChain:
#CHANGED MADE
    def __init__(self, temperature, modelname, username, password):
        llm = create_llm(temperature, modelname, username, password)
        #llm = load_ollama_model()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt)

    def run(self, user_input, chat_history):
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history} ,stop=["Human:"])["text"]