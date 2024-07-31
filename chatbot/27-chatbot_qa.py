# requirements
"""
langchain-openai
langchain
python-dotenv
langchain-community
streamlit
"""

# env variables
"""
 LANGCHAIN_API_KEY
 OPENAI_API_KEY
"""

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.environ.get('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_PROJECT'] = os.environ.get('LANGCHAIN_PROJECT')

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# title of the app
st.title("QA ChatBot with OpenAI")

# side bar
st.sidebar.title("Setting")
apikey = st.sidebar.text_input("Enter your Open AI API Key:", type="password")

# Drop down to select various Open AI models
llm = st.sidebar.selectbox("select model", ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask questions")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, apikey, llm, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the Open AI API key in the side bar.")
else:
    st.write("Please provide question")










