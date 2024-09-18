from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from key import OPENAI_API_KEY, LANGCHAIN_API_KEY
import streamlit as st
import os
from dotenv import load_dotenv
LANGCHAIN_API_KEY="l"
OPENAI_API_KEY="sk"
LANGCHAIN_PROJECT="d"
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful asssitant.please respond to the user qtns"),
        ("user","question:{question}")
    ]
)

st.title("langchain demo with open ai api")
input_text=st.text_input("search the topic u want")
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
