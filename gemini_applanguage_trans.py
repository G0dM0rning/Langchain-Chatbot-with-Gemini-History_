
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from gtts import gTTS
import base64

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{input}"),
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

st.title('Langchain Translator with Gemini + Voice + History')
input_text = st.text_input("Enter text in any language:")
languages = [
    "Urdu", "German", "French", "Spanish", "Arabic",
    "Hindi", "Chinese", "Russian", "Turkish", "Japanese"
]
selected_language = st.selectbox("Select language to translate to:", languages)

if input_text and selected_language:
    response = chain.invoke({
        "input_language": selected_language,
        "output_language": selected_language,
        "input": input_text
    })
    st.markdown(f"### Translated ({selected_language}):")
    st.write(response)

    # Save to history
    with open("translation_history.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Input: {input_text}\nOutput ({selected_language}): {response}\n\n")

    # Text-to-speech
    tts = gTTS(text=response, lang='en')
    tts.save("output.mp3")
    audio_file = open("output.mp3", "rb")
    audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    st.audio(audio_bytes, format='audio/mp3')
