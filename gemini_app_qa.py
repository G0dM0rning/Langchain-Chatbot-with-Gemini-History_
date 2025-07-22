
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Load .env if present
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot"),
    ("human", "Question:{question}")
])

st.title('Langchain Chatbot with Gemini + History')
input_text = st.text_input("Enter your question here")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)
    # Log history
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Q: {input_text}\nA: {response}\n\n")
