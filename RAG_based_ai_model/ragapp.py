import os
import streamlit as st
langchainapikey="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"
fireworks="fw_3ZXFYaWXy9MNCZLKkXr9c2SK"
from transformers import AutoTokenizer, AutoModel
import torch


from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


os.environ["LANGCHAIN_API_KEY"]=langchainapikey
os.environ["LANGCHAIN_TRACING"]="true"
os.environ["FIREWORKS_API_KEY"]=fireworks
from langchain_fireworks import ChatFireworks
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
import bs4
from langchain import hub
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import SoupStrainer as soup
#load data
loader=WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=soup(
            class_=("post-content","post-title","post-header")
        )
    ),
)   
docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.title("rag llm")
input_text=st.text_input("ask me something abt LLM Powered Autonomous Agents")


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.write(rag_chain.invoke(input_text))
